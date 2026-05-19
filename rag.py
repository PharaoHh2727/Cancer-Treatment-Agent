"""
医学RAG模块
使用Chroma向量数据库存储和检索CSCO诊疗指南
"""
import re
import os
import gc
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple
import config

class MedicalRAG:
    """医学知识库RAG"""
    
    def __init__(self, cancer_type: str, verbose: bool = True):
        self.cancer_type = cancer_type.upper()
        self.verbose = verbose
        self.vecdb = None
        self.init_error = None
        self._init_vector_db()

    def _resolve_page_num(self,metadata: Dict) -> str:
        """把 PyPDFLoader 的 0-based page 转成用户可读的 1-based 页码。"""
        page = metadata.get('page')
        if page is not None:
            try:
                return str(int(page) + 1)
            except (TypeError, ValueError):
                pass

        page_num = metadata.get('page_num')
        if page_num is not None:
            try:
                return str(int(page_num))
            except (TypeError, ValueError):
                return str(page_num)

        return "页码未知"

    def _resolve_source_name(self,metadata: Dict) -> str:
        source = metadata.get('source') or metadata.get('file_path') or "CSCO指南"
        source_name = Path(str(source)).name
        return source_name or str(source)

    def _clean_extracted_text(self, text: str) -> str:
        """清理PDF/OCR抽出的文本，避免空字符破坏中文检索和规则过滤。"""
        text = (text or "").replace("\x00", "")
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _keywords(self,query_text: str) -> List[str]:
        keywords = re.findall(r'[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}', query_text or '')
        stopwords = {'BRCA', 'BLCA', 'LUAD', 'TNM', 'Stage'}
        return [kw.lower() for kw in keywords if kw and kw not in stopwords]

    def _is_reference_like_text(self, text: str) -> bool:
        """过滤指南参考文献页，避免把文献列表当成治疗推荐证据。"""
        text = self._clean_extracted_text(text)
        head = text[:300]
        if re.search(r"参考文献|References?", head, flags=re.I):
            return True

        doi_count = len(re.findall(r"\bdoi\b|10\.\d{4,9}/", text, flags=re.I))
        pubmed_count = len(re.findall(r"\bPMID\b|PubMed", text, flags=re.I))
        bracket_ref_count = len(re.findall(r"\[\s*\d+\s*\]|\|\s*\d+\s*\]|\[\s*\d+\s*\||\(\d+\)", text))
        numbered_ref_line_count = len(re.findall(r"(?:^|\n)\s*\[\s*\d+\s*\]", text))
        journal_year_count = len(re.findall(r"\b(?:19|20)\d{2}\s*,\s*\d+\s*(?:\(\s*\d+\s*\))?\s*:\s*\d+", text))
        et_al_count = len(re.findall(r"\bet\s+al\.?", text, flags=re.I))
        citation_author_count = len(re.findall(r"\b[A-Z][A-Z\-]{2,}\s+[A-Z]{1,3}\b", text))
        journal_count = len(re.findall(
            r"\b(?:J|Clin|Oncol|Urol|Eur|Cancer|Cancers|Radiol|Abdom|World|Int|Natl|Compr|Netw)\b",
            text,
            flags=re.I,
        ))
        treatment_hint_count = len(re.findall(
            r"治疗|推荐|方案|一线|二线|辅助|新辅助|化疗|放疗|免疫|靶向|手术|证据级别|适应证",
            text,
        ))
        # 参考文献页通常 DOI/PMID/编号密集；若同一chunk里治疗推荐词很多，则保守保留。
        reference_score = 0
        reference_score += doi_count * 2
        reference_score += pubmed_count * 2
        reference_score += bracket_ref_count
        reference_score += numbered_ref_line_count * 2
        reference_score += journal_year_count * 4
        reference_score += et_al_count * 3
        reference_score += min(citation_author_count, 8)
        reference_score += min(journal_count, 8)

        return (
            doi_count >= 3
            or pubmed_count >= 3
            or bracket_ref_count >= 8
            or numbered_ref_line_count >= 3
            or journal_year_count >= 2
            or et_al_count >= 2
            or reference_score >= 12
        ) and treatment_hint_count < 6

    def _filter_reference_docs(self, docs) -> List:
        filtered_docs = []
        removed = 0
        for doc in docs or []:
            content = self._clean_extracted_text(getattr(doc, "page_content", ""))
            if not content:
                continue
            if self._is_reference_like_text(content):
                removed += 1
                continue
            doc.page_content = content
            filtered_docs.append(doc)

        if removed and self.verbose:
            print(f"  已过滤参考文献类CSCO chunk: {removed}个")
        return filtered_docs

    def _get_guide_path(self) -> Tuple[Path, str]:
        guide_file = config.CANCER_TYPES.get(self.cancer_type)
        if not guide_file:
            return None, f"未找到{self.cancer_type}对应的CSCO指南配置"

        guide_path = config.GUIDE_PDF_DIR / guide_file
        if not guide_path.exists():
            return guide_path, f"CSCO指南文件不存在: {guide_path}"
        return guide_path, ""

    def _is_chroma_schema_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return (
            "no such table: tenants" in message
            or "no such table: collections" in message
            or "could not connect to tenant" in message
            or "tenant" in message and "not found" in message
        )

    def _safe_clear_vector_dir(self, db_path: Path):
        db_path = Path(db_path).resolve()
        root = Path(config.RAG_DB_DIR).resolve()
        if db_path == root or root not in db_path.parents:
            raise ValueError(f"拒绝清理非RAG_DB_DIR下的路径: {db_path}")
        if not db_path.exists():
            return

        self.vecdb = None
        gc.collect()

        if self.verbose:
            print(f"  清理损坏的Chroma知识库目录: {db_path}")

        for attempt in range(3):
            try:
                shutil.rmtree(db_path)
                return
            except OSError as e:
                if self.verbose:
                    print(f"  第{attempt + 1}次清理Chroma目录失败: {e}")
                time.sleep(0.5 * (attempt + 1))

        # 如果目录仍被Chroma/sqlite临时文件占用，先改名隔离，让原路径可以重新建库。
        backup_path = db_path.with_name(f"{db_path.name}.corrupt_{int(time.time())}")
        try:
            db_path.rename(backup_path)
            if self.verbose:
                print(f"  已将损坏Chroma目录隔离为: {backup_path}")
            try:
                shutil.rmtree(backup_path)
            except OSError as e:
                if self.verbose:
                    print(f"  隔离目录暂未删除，可稍后手动清理: {backup_path}; {e}")
        except OSError as e:
            raise OSError(f"无法清理或隔离Chroma目录 {db_path}: {e}") from e

    def _query_pdf_fallback(self, query_text: str, k: int) -> str:
        """离线兜底检索：直接扫描本地CSCO PDF页文本，不依赖embedding和HuggingFace网络。"""
        guide_path, guide_error = self._get_guide_path()
        if guide_error:
            return f"CSCO RAG检索失败: {guide_error}"

        pages = []

        # 优先使用pypdf/PyPDF2；如果没有，再尝试LangChain的PyPDFLoader。
        try:
            try:
                from pypdf import PdfReader
            except Exception:
                from PyPDF2 import PdfReader

            reader = PdfReader(str(guide_path))
            for page_idx, page in enumerate(reader.pages):
                text = self._clean_extracted_text(page.extract_text() or "")
                if text:
                    pages.append((page_idx + 1, text))
        except Exception:
            try:
                from langchain_community.document_loaders import PyPDFLoader

                docs = PyPDFLoader(str(guide_path)).load()
                for doc in docs:
                    text = self._clean_extracted_text(doc.page_content or "")
                    if not text:
                        continue
                    page_num = self._resolve_page_num(doc.metadata)
                    try:
                        page_num = int(page_num)
                    except (TypeError, ValueError):
                        page_num = len(pages) + 1
                    pages.append((page_num, text))
            except Exception as e:
                return f"CSCO RAG检索失败: 向量库不可用，PDF兜底检索也失败: {e}"

        if not pages:
            return f"CSCO RAG检索失败: 本地PDF未提取到可检索文本: {guide_path}"

        keywords = self._keywords(query_text)
        # 指南中更可能出现这些中文词。即使query关键词不命中，也用这些词找治疗相关页。
        treatment_terms = [
            "治疗", "方案", "推荐", "分期", "辅助", "新辅助", "一线", "二线",
            "化疗", "放疗", "免疫", "靶向", "内分泌", "手术", "复发", "转移"
        ]

        candidates = []
        non_reference_pages = [(page_num, text) for page_num, text in pages if not self._is_reference_like_text(text)]
        scored_pages = non_reference_pages or pages

        for page_num, text in scored_pages:
            lower_text = text.lower()
            score = 0
            score += sum(lower_text.count(keyword) * 3 for keyword in keywords)
            score += sum(text.count(term) for term in treatment_terms)

            if score > 0:
                candidates.append((score, page_num, text[:1200]))

        if not candidates:
            # 最后保证不返回空：如果PDF有文本，返回前k个非空页，并明确说明是兜底结果。
            fallback_pages = non_reference_pages or pages
            candidates = [(1, page_num, text[:1200]) for page_num, text in fallback_pages[:k]]
            if self.verbose:
                print("  PDF关键词未命中，返回前置非空页作为CSCO兜底证据")

        candidates.sort(key=lambda item: item[0], reverse=True)
        results = []
        for i, (_, page_num, content) in enumerate(candidates[:k], 1):
            results.append(f"【CSCO-{i} | {guide_path.name} | 第{page_num}页】\n{content}")

        return "\n\n".join(results)

    def _create_embedding_model(self):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )

    def _collection_count(self) -> int:
        if self.vecdb is None:
            return 0

        try:
            return int(self.vecdb._collection.count())
        except Exception:
            pass

        try:
            data = self.vecdb.get(include=[])
            return len(data.get('ids', []))
        except Exception:
            return 0

    def _collection_has_dirty_text(self) -> bool:
        """检测旧版Chroma库中是否存在带空字符的OCR文本。"""
        if self.vecdb is None:
            return False

        try:
            data = self.vecdb.get(include=["documents"], limit=20)
        except Exception:
            return False

        return any("\x00" in (doc or "") for doc in data.get("documents") or [])

    def _init_vector_db(self):
        """初始化向量数据库"""
        try:
            if not getattr(config, "RAG_USE_VECTOR_DB", False):
                self.init_error = "向量库模式未启用，将使用本地PDF离线检索"
                if self.verbose:
                    print(f"  {self.init_error}")
                self.vecdb = None
                return

            from langchain_chroma import Chroma
            
            # 初始化embedding模型
            embedding_model = self._create_embedding_model()
            
            # 检查知识库是否存在
            db_path = config.RAG_DB_DIR / self.cancer_type
            
            if db_path.exists():
                # 加载已有知识库
                try:
                    self.vecdb = Chroma(
                        persist_directory=str(db_path),
                        embedding_function=embedding_model
                    )
                except Exception as e:
                    if not self._is_chroma_schema_error(e):
                        raise
                    if self.verbose:
                        print(f"  Chroma知识库结构损坏或版本不兼容: {e}")
                    self.vecdb = None
                    self._safe_clear_vector_dir(db_path)
                    self._create_vector_db(db_path, embedding_model)
                    return

                if self.verbose:
                    print(f"  已加载知识库: {db_path}，文档数: {self._collection_count()}")

                if self._collection_count() == 0:
                    if self.verbose:
                        print(f"  知识库为空，重新从CSCO PDF构建: {db_path}")
                    self.vecdb = None
                    self._safe_clear_vector_dir(db_path)
                    self._create_vector_db(db_path, embedding_model)
                elif self._collection_has_dirty_text():
                    if self.verbose:
                        print(f"  检测到旧版知识库含OCR空字符，重新从CSCO PDF构建: {db_path}")
                    self.vecdb = None
                    self._safe_clear_vector_dir(db_path)
                    self._create_vector_db(db_path, embedding_model)
            else:
                # 知识库不存在，创建新的
                self._create_vector_db(db_path, embedding_model)
                
        except Exception as e:
            self.init_error = f"初始化向量库失败: {e}"
            if self.verbose:
                print(f"  {self.init_error}")
            self.vecdb = None
    
    def _create_vector_db(self, db_path: Path, embedding_model):
        """创建向量数据库"""
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_chroma import Chroma
            
            # 获取指南PDF
            guide_path, guide_error = self._get_guide_path()
            if guide_error:
                self.init_error = guide_error
                if self.verbose:
                    print(f"  {self.init_error}")
                return
            
            # 加载PDF
            if self.verbose:
                print(f"  加载指南: {guide_path}")
            
            loader = PyPDFLoader(str(guide_path))
            documents = loader.load()
            for doc in documents:
                doc.page_content = self._clean_extracted_text(doc.page_content)
            documents = [doc for doc in documents if (doc.page_content or '').strip()]
            if not documents:
                self.init_error = f"CSCO指南PDF未提取到可检索文本，无法创建知识库: {guide_path}"
                if self.verbose:
                    print(f"  {self.init_error}")
                return
            
            # 添加页码到 metadata
            for doc in documents:
                doc.metadata['source'] = guide_path.name
                doc.metadata['page_num'] = self._resolve_page_num(doc.metadata)
            
            # 分割文本
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            for doc in splits:
                doc.page_content = self._clean_extracted_text(doc.page_content)
            splits = [doc for doc in splits if (doc.page_content or '').strip()]
            splits = [doc for doc in splits if not self._is_reference_like_text(doc.page_content)]
            if not splits:
                self.init_error = f"CSCO指南文本切分后没有有效chunk，无法创建知识库: {guide_path}"
                if self.verbose:
                    print(f"  {self.init_error}")
                return
            
            # 创建向量库
            db_path.mkdir(parents=True, exist_ok=True)
            self.vecdb = Chroma.from_documents(
                documents=splits,
                embedding=embedding_model,
                persist_directory=str(db_path)
            )
            
            if self.verbose:
                print(f"  创建知识库成功，文档数: {len(splits)}")
            self.init_error = None
                
        except Exception as e:
            if self._is_chroma_schema_error(e):
                try:
                    self.vecdb = None
                    self._safe_clear_vector_dir(db_path)
                    self.vecdb = Chroma.from_documents(
                        documents=splits,
                        embedding=embedding_model,
                        persist_directory=str(db_path)
                    )
                    if self.verbose:
                        print(f"  清理损坏目录后创建知识库成功，文档数: {len(splits)}")
                    self.init_error = None
                    return
                except Exception as retry_error:
                    self.init_error = f"清理损坏目录后创建知识库仍失败: {retry_error}"
            else:
                self.init_error = f"创建知识库失败: {e}"
            if self.verbose:
                print(f"  {self.init_error}")

    def _rebuild_vector_db(self):
        try:
            embedding_model = self._create_embedding_model()
            db_path = config.RAG_DB_DIR / self.cancer_type
            self._create_vector_db(db_path, embedding_model)
        except Exception as e:
            self.init_error = f"重建知识库失败: {e}"
            if self.verbose:
                print(f"  {self.init_error}")

    def _format_docs(self, docs) -> str:
        results = []
        for i, doc in enumerate(docs, 1):
            content = self._clean_extracted_text(doc.page_content)
            if not content:
                continue

            page_num = self._resolve_page_num(doc.metadata)
            source = self._resolve_source_name(doc.metadata)
            page_label = f"第{page_num}页" if page_num != "页码未知" else page_num
            results.append(f"【CSCO-{i} | {source} | {page_label}】\n{content}")

        return '\n\n'.join(results)

    def _make_doc(self, page_content: str, metadata: Dict):
        class _Doc:
            def __init__(self, content, meta):
                self.page_content = content
                self.metadata = meta

        return _Doc(page_content, metadata or {})

    def _doc_key(self, doc) -> tuple:
        metadata = getattr(doc, "metadata", {}) or {}
        content = self._clean_extracted_text(getattr(doc, "page_content", ""))
        return (
            self._resolve_source_name(metadata),
            self._resolve_page_num(metadata),
            content[:120],
        )

    def _keyword_fallback_docs_from_collection(self, query_text: str, k: int, seen_keys: set | None = None) -> List:
        """从Chroma已存文本中做关键词兜底检索，返回可继续合并的doc列表。"""
        if self.vecdb is None:
            return []

        try:
            data = self.vecdb.get(include=["documents", "metadatas"])
        except Exception as e:
            if self.verbose:
                print(f"  Chroma关键词兜底读取失败: {e}")
            return []

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        keywords = self._keywords(query_text)
        fallback_terms = ['治疗', '方案', '推荐', '分期', '辅助', '一线', '二线', '化疗', '免疫', '靶向']
        candidates = []
        seen_keys = seen_keys or set()

        for idx, content in enumerate(documents):
            content = self._clean_extracted_text(content)
            if not content:
                continue
            if self._is_reference_like_text(content):
                continue
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            doc = self._make_doc(content, metadata)
            if self._doc_key(doc) in seen_keys:
                continue

            lower_content = content.lower()
            score = sum(lower_content.count(keyword) for keyword in keywords)
            if score == 0:
                score = sum(content.count(term) for term in fallback_terms)

            if score > 0:
                candidates.append((score, doc))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in candidates[:k]]

    def _keyword_fallback_from_collection(self, query_text: str, k: int) -> str:
        """当向量检索返回空时，从Chroma已存文本中做关键词兜底检索。"""
        docs = self._keyword_fallback_docs_from_collection(query_text, k)
        return self._format_docs(docs)

    def _first_nonempty_docs_from_collection(self, k: int, seen_keys: set | None = None) -> List:
        """最后兜底：collection非空但检索不命中时，返回前k个非空CSCO文本块。"""
        if self.vecdb is None:
            return []

        try:
            data = self.vecdb.get(include=["documents", "metadatas"], limit=max(k * 5, 10))
        except Exception as e:
            if self.verbose:
                print(f"  Chroma前置文本兜底读取失败: {e}")
            return []

        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        seen_keys = seen_keys or set()

        docs = []
        for idx, content in enumerate(documents):
            content = self._clean_extracted_text(content)
            if not content:
                continue
            if self._is_reference_like_text(content):
                continue
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            doc = self._make_doc(content, metadata)
            if self._doc_key(doc) in seen_keys:
                continue
            docs.append(doc)
            if len(docs) >= k:
                break

        return docs

    def _first_nonempty_from_collection(self, k: int) -> str:
        docs = self._first_nonempty_docs_from_collection(k)
        return self._format_docs(docs)

    def _supplement_docs(self, docs: List, query_text: str, k: int) -> List:
        docs = list(docs or [])[:k]
        if len(docs) >= k:
            return docs

        seen_keys = {self._doc_key(doc) for doc in docs}
        need = k - len(docs)
        fallback_docs = self._keyword_fallback_docs_from_collection(query_text, need, seen_keys)
        if fallback_docs:
            if self.verbose:
                print(f"  向量检索结果少于top_k，已用关键词兜底补齐{len(fallback_docs)}条CSCO证据")
            docs.extend(fallback_docs)
            seen_keys.update(self._doc_key(doc) for doc in fallback_docs)

        if len(docs) >= k:
            return docs[:k]

        need = k - len(docs)
        fallback_docs = self._first_nonempty_docs_from_collection(need, seen_keys)
        if fallback_docs:
            if self.verbose:
                print(f"  向量检索和关键词兜底仍少于top_k，已补充{len(fallback_docs)}条非空CSCO证据")
            docs.extend(fallback_docs)

        return docs[:k]
    
    def query(self, query_text: str, k: int = None) -> str:
        """查询知识库"""
        k = k or config.RAG_TOP_K

        if self.vecdb is None:
            pdf_result = self._query_pdf_fallback(query_text, k)
            if "【CSCO-" in pdf_result:
                if self.verbose:
                    print("  已使用本地PDF离线检索返回CSCO证据")
                return pdf_result
            return f"CSCO RAG检索失败: {self.init_error or '知识库未初始化，无法查询'}; {pdf_result}"

        if self._collection_count() == 0:
            if self.verbose:
                print("  Chroma collection文档数为0，尝试重建知识库")
            self._rebuild_vector_db()
            if self.vecdb is None or self._collection_count() == 0:
                return f"CSCO RAG检索失败: 知识库为空，重建后仍无文档。{self.init_error or ''}"

        try:
            docs = self.vecdb.similarity_search(query_text, k=max(k * 5, k + 8))
        except Exception:
            retriever = self.vecdb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": max(k * 5, k + 8)}
            )
            docs = retriever.invoke(query_text)

        docs = self._supplement_docs(self._filter_reference_docs(docs), query_text, k)

        if not docs:
            fallback = self._keyword_fallback_from_collection(query_text, k)
            if fallback:
                if self.verbose:
                    print("  向量检索为空，已使用Chroma文本关键词兜底结果")
                return fallback

            fallback = self._first_nonempty_from_collection(k)
            if fallback:
                if self.verbose:
                    print("  向量检索为空，已使用Chroma前置非空文本兜底结果")
                return fallback

            if self.verbose:
                print("  向量检索为空且关键词兜底为空，尝试重建知识库后重查")
            self._rebuild_vector_db()
            if self.vecdb is not None and self._collection_count() > 0:
                docs = self.vecdb.similarity_search(query_text, k=max(k * 5, k + 8))
                docs = self._supplement_docs(self._filter_reference_docs(docs), query_text, k)
                if docs:
                    return self._format_docs(docs)
                fallback = self._keyword_fallback_from_collection(query_text, k)
                if fallback:
                    return fallback
                fallback = self._first_nonempty_from_collection(k)
                if fallback:
                    return fallback

            return f"CSCO RAG检索失败: 已加载知识库，但未检索到相关内容。query={query_text}; collection_count={self._collection_count()}"

        result = self._format_docs(docs)

        if not result:
            fallback = self._keyword_fallback_from_collection(query_text, k)
            if fallback:
                return fallback
            fallback = self._first_nonempty_from_collection(k)
            if fallback:
                return fallback
            return f"CSCO RAG检索失败: 检索结果为空文本。query={query_text}; collection_count={self._collection_count()}"

        return result

    def _has_valid_csco_rag(self,guide_content: str) -> bool:
        if not guide_content:
            return False
        failure_markers = [
            "CSCO RAG检索失败",
            "知识库未初始化",
            "无法查询",
            "查询失败",
            "未检索到相关内容",
            "检索结果为空文本",
        ]
        return "【CSCO-" in guide_content and not any(marker in guide_content for marker in failure_markers)

def create_rag(cancer_type: str, verbose: bool = True) -> MedicalRAG:
    """创建医学RAG实例"""
    return MedicalRAG(cancer_type, verbose=verbose)

if __name__ == "__main__":
    # 测试
    rag = create_rag("BRCA")
    result = rag.query("乳腺癌II期治疗方案")
    print(result)
