# Refactored from kokoro-tts CLI script for API and CLI use
import os
import numpy as np
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import soundfile as sf
from kokoro_onnx import Kokoro
import pymupdf4llm
import fitz
import re

# --- Utility Functions ---
def validate_language(lang, kokoro):
    try:
        supported_languages = set(kokoro.get_languages())
        if lang not in supported_languages:
            raise ValueError(f"Unsupported language: {lang}")
        return lang
    except Exception as e:
        raise ValueError(f"Error getting supported languages: {e}")

def validate_voice(voice, kokoro):
    try:
        supported_voices = set(kokoro.get_voices())
        if ',' in voice:
            voices = []
            weights = []
            for pair in voice.split(','):
                if ':' in pair:
                    v, w = pair.strip().split(':')
                    voices.append(v.strip())
                    weights.append(float(w.strip()))
                else:
                    voices.append(pair.strip())
                    weights.append(50.0)
            if len(voices) != 2:
                raise ValueError("voice blending needs two comma separated voices")
            for v in voices:
                if v not in supported_voices:
                    raise ValueError(f"Unsupported voice: {v}")
            total = sum(weights)
            if total != 100:
                weights = [w * (100/total) for w in weights]
            style1 = kokoro.get_voice_style(voices[0])
            style2 = kokoro.get_voice_style(voices[1])
            blend = np.add(style1 * (weights[0]/100), style2 * (weights[1]/100))
            return blend
        if voice not in supported_voices:
            raise ValueError(f"Unsupported voice: {voice}")
        return voice
    except Exception as e:
        raise ValueError(f"Error getting supported voices: {e}")

def extract_chapters_from_epub(epub_file, debug=False):
    book = epub.read_epub(epub_file)
    chapters = []
    def get_chapter_content(soup, start_id, next_id=None):
        content = []
        start_elem = soup.find(id=start_id)
        if not start_elem:
            return ""
        if start_elem.name in ['h1', 'h2', 'h3', 'h4']:
            current = start_elem.find_next_sibling()
        else:
            current = start_elem
        while current:
            if next_id and current.get('id') == next_id:
                break
            if current.name in ['h1', 'h2', 'h3'] and 'chapter' in current.get_text().lower():
                break
            content.append(current.get_text())
            current = current.find_next_sibling()
        return '\n'.join(content).strip()
    def process_toc_items(items, depth=0):
        processed = []
        for i, item in enumerate(items):
            if isinstance(item, tuple):
                section_title, section_items = item
                processed.extend(process_toc_items(section_items, depth + 1))
            elif isinstance(item, epub.Link):
                href_parts = item.href.split('#')
                file_name = href_parts[0]
                fragment_id = href_parts[1] if len(href_parts) > 1 else None
                doc = next((doc for doc in book.get_items_of_type(ITEM_DOCUMENT) 
                          if doc.file_name.endswith(file_name)), None)
                if doc:
                    content = doc.get_content().decode('utf-8')
                    soup = BeautifulSoup(content, "html.parser")
                    if not fragment_id:
                        text_content = soup.get_text().strip()
                    else:
                        next_item = items[i + 1] if i + 1 < len(items) else None
                        next_fragment = None
                        if isinstance(next_item, epub.Link):
                            next_href_parts = next_item.href.split('#')
                            if next_href_parts[0] == file_name and len(next_href_parts) > 1:
                                next_fragment = next_href_parts[1]
                        text_content = get_chapter_content(soup, fragment_id, next_fragment)
                    if text_content:
                        chapters.append({
                            'title': item.title,
                            'content': text_content,
                            'order': len(processed) + 1
                        })
                        processed.append(item)
        return processed
    process_toc_items(book.toc)
    if not chapters:
        docs = sorted(
            book.get_items_of_type(ITEM_DOCUMENT),
            key=lambda x: x.file_name
        )
        for doc in docs:
            content = doc.get_content().decode('utf-8')
            soup = BeautifulSoup(content, "html.parser")
            chapter_divs = soup.find_all(['h1', 'h2', 'h3'], class_=lambda x: x and 'chapter' in x.lower())
            if not chapter_divs:
                chapter_divs = soup.find_all(lambda tag: tag.name in ['h1', 'h2', 'h3'] and 
                                          ('chapter' in tag.get_text().lower() or
                                           'book' in tag.get_text().lower()))
            if chapter_divs:
                for i, div in enumerate(chapter_divs):
                    title = div.get_text().strip()
                    content = ''
                    for tag in div.find_next_siblings():
                        if tag.name in ['h1', 'h2', 'h3'] and (
                            'chapter' in tag.get_text().lower() or
                            'book' in tag.get_text().lower()):
                            break
                        content += tag.get_text() + '\n'
                    if content.strip():
                        chapters.append({
                            'title': title,
                            'content': content.strip(),
                            'order': len(chapters) + 1
                        })
            else:
                text_content = soup.get_text().strip()
                if text_content:
                    title_tag = soup.find(['h1', 'h2', 'title'])
                    title = title_tag.get_text().strip() if title_tag else f"Chapter {len(chapters) + 1}"
                    if title.lower() not in ['copy', 'copyright', 'title page', 'cover']:
                        chapters.append({
                            'title': title,
                            'content': text_content,
                            'order': len(chapters) + 1
                        })
    return chapters

def chunk_text(text, initial_chunk_size=1000):
    sentences = text.replace('\n', ' ').split('.')
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_size = initial_chunk_size
    for sentence in sentences:
        if not sentence.strip():
            continue
        sentence = sentence.strip() + '.'
        sentence_size = len(sentence)
        if sentence_size > chunk_size:
            words = sentence.split()
            current_piece = []
            current_piece_size = 0
            for word in words:
                word_size = len(word) + 1
                if current_piece_size + word_size > chunk_size:
                    if current_piece:
                        chunks.append(' '.join(current_piece).strip() + '.')
                    current_piece = [word]
                    current_piece_size = word_size
                else:
                    current_piece.append(word)
                    current_piece_size += word_size
            if current_piece:
                chunks.append(' '.join(current_piece).strip() + '.')
            continue
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(sentence)
        current_size += sentence_size
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

class PdfParser:
    def __init__(self, pdf_path: str, debug: bool = False, min_chapter_length: int = 50):
        self.pdf_path = pdf_path
        self.chapters = []
        self.debug = debug
        self.min_chapter_length = min_chapter_length
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    def get_chapters(self):
        if self.get_chapters_from_toc():
            return self.chapters
        self.chapters = self.get_chapters_from_markdown()
        return self.chapters
    def get_chapters_from_toc(self):
        doc = None
        try:
            doc = fitz.open(self.pdf_path)
            toc = doc.get_toc()
            if not toc:
                return False
            seen_pages = set()
            chapter_markers = []
            for level, title, page in toc:
                if level == 1:
                    title = self._clean_title(title)
                    if title and page not in seen_pages:
                        chapter_markers.append((title, page))
                        seen_pages.add(page)
            if not chapter_markers:
                return False
            for i, (title, start_page) in enumerate(chapter_markers):
                end_page = (chapter_markers[i + 1][1] - 1 
                           if i < len(chapter_markers) - 1 
                           else doc.page_count)
                chapter_text = self._extract_chapter_text(doc, start_page - 1, end_page)
                if len(chapter_text.strip()) > self.min_chapter_length:
                    self.chapters.append({
                        'title': title,
                        'content': chapter_text,
                        'order': i + 1
                    })
            return bool(self.chapters)
        except Exception:
            return False
        finally:
            if doc:
                doc.close()
    def get_chapters_from_markdown(self):
        chapters = []
        try:
            def progress(current, total):
                pass
            md_text = pymupdf4llm.to_markdown(
                self.pdf_path,
                show_progress=False,
                progress_callback=progress
            )
            md_text = self._clean_markdown(md_text)
            current_chapter = None
            current_text = []
            chapter_count = 0
            for line in md_text.split('\n'):
                if line.startswith('#'):
                    if current_chapter and current_text:
                        chapter_text = ''.join(current_text)
                        if len(chapter_text.strip()) > self.min_chapter_length:
                            chapters.append({
                                'title': current_chapter,
                                'content': chapter_text,
                                'order': chapter_count
                            })
                    chapter_count += 1
                    current_chapter = f"Chapter {chapter_count}_{line.lstrip('#').strip()}"
                    current_text = []
                else:
                    if current_chapter is not None:
                        current_text.append(line + '\n')
            if current_chapter and current_text:
                chapter_text = ''.join(current_text)
                if len(chapter_text.strip()) > self.min_chapter_length:
                    chapters.append({
                        'title': current_chapter,
                        'content': chapter_text,
                        'order': chapter_count
                    })
            return chapters
        except Exception:
            return chapters
    def _clean_title(self, title: str) -> str:
        return title.strip().replace('\u200b', ' ')
    def _clean_markdown(self, text: str) -> str:
        text = text.replace('-', '')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    def _extract_chapter_text(self, doc, start_page: int, end_page: int) -> str:
        chapter_text = []
        for page_num in range(start_page, end_page):
            try:
                page = doc[page_num]
                text = page.get_text()
                chapter_text.append(text)
            except Exception:
                continue
        return '\n'.join(chapter_text)

def process_chunk_sequential(chunk: str, kokoro: Kokoro, voice: str, speed: float, lang: str, 
                           retry_count=0, debug=False):
    try:
        samples, sample_rate = kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
        return samples, sample_rate
    except Exception as e:
        error_msg = str(e)
        if "index 510 is out of bounds" in error_msg:
            current_size = len(chunk)
            new_size = int(current_size * 0.6)
            words = chunk.split()
            current_piece = []
            current_size = 0
            pieces = []
            for word in words:
                word_size = len(word) + 1
                if current_size + word_size > new_size:
                    if current_piece:
                        pieces.append(' '.join(current_piece).strip())
                    current_piece = [word]
                    current_size = word_size
                else:
                    current_piece.append(word)
                    current_size += word_size
            if current_piece:
                pieces.append(' '.join(current_piece).strip())
            all_samples = []
            last_sample_rate = None
            for piece in pieces:
                samples, sr = process_chunk_sequential(piece, kokoro, voice, speed, lang, 
                                                     retry_count + 1, debug)
                if samples is not None:
                    all_samples.extend(samples)
                    last_sample_rate = sr
            if all_samples:
                return all_samples, last_sample_rate
        raise RuntimeError(f"Error processing chunk: {e}")

def merge_chunks_to_chapters(split_output_dir, format="wav"):
    if not os.path.exists(split_output_dir):
        raise FileNotFoundError(f"Directory {split_output_dir} does not exist.")
    chapter_dirs = sorted([d for d in os.listdir(split_output_dir) 
                          if d.startswith("chapter_") and os.path.isdir(os.path.join(split_output_dir, d))])
    if not chapter_dirs:
        raise FileNotFoundError(f"No chapter directories found in {split_output_dir}")
    used_titles = set()
    for chapter_dir in chapter_dirs:
        chapter_path = os.path.join(split_output_dir, chapter_dir)
        chunk_files = sorted([f for f in os.listdir(chapter_path) 
                            if f.startswith("chunk_") and f.endswith(f".{format}")])
        if not chunk_files:
            continue
        chapter_title = chapter_dir
        info_file = os.path.join(chapter_path, "info.txt")
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("Title:"):
                        chapter_title = line.replace("Title:", "").strip()
                        break
        safe_title = "".join(c for c in chapter_title if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_title or safe_title in used_titles:
            merged_file = os.path.join(split_output_dir, f"{chapter_dir}.{format}")
        else:
            merged_file = os.path.join(split_output_dir, f"{safe_title}.{format}")
            used_titles.add(safe_title)
        all_samples = []
        sample_rate = None
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chapter_path, chunk_file)
            try:
                data, sr = sf.read(chunk_path)
                if len(data) == 0:
                    continue
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    continue
                all_samples.extend(data)
            except Exception:
                continue
        if all_samples:
            all_samples = np.array(all_samples)
            sf.write(merged_file, all_samples, sample_rate) 