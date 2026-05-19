# PubMed文献搜索工具

import csv
import os
import re
import html
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


# PubMed E-utilities API配置
PUBMED_API_BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
PUBMED_API_KEY = os.getenv('PUBMED_API_KEY', '')  # API Key可选；不设置也可以低频检索


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


DEFAULT_RECENT_YEARS = max(1, _env_int('PUBMED_RECENT_YEARS', 5))
DEFAULT_MIN_IMPACT_FACTOR = max(0.0, _env_float('PUBMED_MIN_IMPACT_FACTOR', 10.0))
JOURNAL_METRICS_CSV = os.getenv('PUBMED_JOURNAL_METRICS_CSV', '').strip()

CANCER_QUERY_MAP = {
    "BRCA": '"Breast Neoplasms"[Mesh] OR "breast cancer"[Title/Abstract] OR "breast carcinoma"[Title/Abstract]',
    "BLCA": '"Urinary Bladder Neoplasms"[Mesh] OR "bladder cancer"[Title/Abstract] OR "bladder carcinoma"[Title/Abstract] OR "urothelial carcinoma"[Title/Abstract] OR "urothelial cancer"[Title/Abstract] OR "muscle invasive bladder cancer"[Title/Abstract] OR "non muscle invasive bladder cancer"[Title/Abstract] OR MIBC[Title/Abstract] OR NMIBC[Title/Abstract]',
    "LUAD": '"Lung Neoplasms"[Mesh] OR "lung adenocarcinoma"[Title/Abstract] OR "non-small cell lung cancer"[Title/Abstract] OR NSCLC[Title/Abstract]',
    "HNSC": '"Squamous Cell Carcinoma of Head and Neck"[Mesh] OR "head and neck squamous cell carcinoma"[Title/Abstract]',
    "COADREAD": '"Colorectal Neoplasms"[Mesh] OR "colorectal cancer"[Title/Abstract] OR "colon cancer"[Title/Abstract] OR "rectal cancer"[Title/Abstract]',
    "UCEC": '"Endometrial Neoplasms"[Mesh] OR "endometrial cancer"[Title/Abstract] OR "endometrial carcinoma"[Title/Abstract]',
}

TREATMENT_TERMS = (
    '"treatment"[Title/Abstract] OR "therapy"[Title/Abstract] OR chemotherapy[Title/Abstract] OR '
    'radiotherapy[Title/Abstract] OR immunotherapy[Title/Abstract] OR "targeted therapy"[Title/Abstract] OR '
    '"endocrine therapy"[Title/Abstract] OR surgery[Title/Abstract] OR neoadjuvant[Title/Abstract] OR '
    'adjuvant[Title/Abstract] OR maintenance[Title/Abstract] OR "first-line"[Title/Abstract] OR '
    '"second-line"[Title/Abstract] OR "clinical trial"[Publication Type] OR guideline[Publication Type]'
)

CANCER_FILTER_PATTERNS = {
    "BRCA": [
        r"\bbreast (cancer|carcinoma|tumou?r|neoplasm|malignan)",
        r"\bbreast neoplasms?\b",
    ],
    "BLCA": [
        r"\bbladder (cancer|carcinoma|tumou?r|neoplasm|malignan)",
        r"\burothelial (carcinoma|cancer|tumou?r|neoplasm|malignan)",
        r"\burinary bladder neoplasms?\b",
        r"\bNMIBC\b",
        r"\bMIBC\b",
        r"\bmUC\b",
    ],
    "LUAD": [
        r"\blung adenocarcinoma\b",
        r"\bnon[- ]small cell lung cancer\b",
        r"\bNSCLC\b",
        r"\blung (cancer|carcinoma|tumou?r|neoplasm|malignan)",
    ],
    "HNSC": [r"\bhead and neck squamous cell carcinoma\b", r"\bHNSCC\b"],
    "COADREAD": [
        r"\bcolorectal (cancer|carcinoma|tumou?r|neoplasm|malignan)",
        r"\bcolon (cancer|carcinoma|tumou?r|neoplasm|malignan)",
        r"\brectal (cancer|carcinoma|tumou?r|neoplasm|malignan)",
    ],
    "UCEC": [r"\bendometrial (cancer|carcinoma|tumou?r|neoplasm|malignan)"],
}

TREATMENT_FILTER_PATTERNS = [
    r"\btreat(ment|ed|ing)?\b",
    r"\btherap(y|ies|eutic)\b",
    r"\bchemotherapy\b",
    r"\bradiotherapy\b",
    r"\bradiation therapy\b",
    r"\bimmunotherapy\b",
    r"\bimmune checkpoint\b",
    r"\btargeted therapy\b",
    r"\bneoadjuvant\b",
    r"\badjuvant\b",
    r"\bmaintenance\b",
    r"\bfirst[- ]line\b",
    r"\bsecond[- ]line\b",
    r"\bsurgery\b",
    r"\bcystectomy\b",
    r"\bTURBT\b",
    r"\bintravesical\b",
    r"\bBCG\b",
    r"\bcisplatin\b",
    r"\bcarboplatin\b",
    r"\bgemcitabine\b",
    r"\bpembrolizumab\b",
    r"\bnivolumab\b",
    r"\batezolizumab\b",
    r"\bavelumab\b",
    r"\benfortumab\b",
    r"\berdafitinib\b",
    r"\btrastuzumab\b",
    r"\bendocrine therapy\b",
    r"\bclinical trial\b",
]

NON_TREATMENT_ONLY_PATTERNS = [
    r"\bdetection\b",
    r"\bdiagnos(is|tic)\b",
    r"\bscreening\b",
    r"\bsurveillance\b",
    r"\bworkflow\b",
    r"\bimaging-guided\b",
    r"\bbiomarker\b",
]

PRECLINICAL_PATTERNS = [
    r"\bmurine\b",
    r"\bmice\b",
    r"\bmouse\b",
    r"\brat\b",
    r"\bcell line\b",
    r"\bin vitro\b",
    r"\bpreclinical\b",
    r"\bxenograft\b",
]

# PubMed does not expose journal impact factor. This built-in list is only a
# fallback for common high-impact clinical/oncology journals; set
# PUBMED_JOURNAL_METRICS_CSV for exact yearly JCR/CiteScore data.
BUILTIN_HIGH_IMPACT_JOURNAL_FLOOR = 10.01
BUILTIN_HIGH_IMPACT_JOURNALS = [
    "New England Journal of Medicine",
    "N Engl J Med",
    "Lancet",
    "The Lancet",
    "JAMA",
    "BMJ",
    "Nature",
    "Science",
    "Cell",
    "Nature Medicine",
    "Nat Med",
    "Lancet Oncology",
    "The Lancet Oncology",
    "Lancet Oncol",
    "Journal of Clinical Oncology",
    "J Clin Oncol",
    "Annals of Oncology",
    "Ann Oncol",
    "JAMA Oncology",
    "JAMA Oncol",
    "Cancer Cell",
    "Cancer Discovery",
    "Cancer Discov",
    "Clinical Cancer Research",
    "Clin Cancer Res",
    "Nature Cancer",
    "Nat Cancer",
    "Nature Reviews Clinical Oncology",
    "Nat Rev Clin Oncol",
    "Molecular Cancer",
    "Mol Cancer",
    "Signal Transduction and Targeted Therapy",
    "Signal Transduct Target Ther",
    "Journal for Immunotherapy of Cancer",
    "J Immunother Cancer",
    "Blood",
    "Leukemia",
    "Gastroenterology",
    "Gut",
    "Hepatology",
    "European Urology",
    "Eur Urol",
    "ESMO Open",
]

_JOURNAL_METRIC_CACHE: Optional[Dict[str, Dict]] = None


def _normalize_journal_name(value: str) -> str:
    value = _clean_text(value).lower()
    value = value.replace('&', ' and ')
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return re.sub(r'^the\s+', '', value)


def _parse_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(',', '')
    match = re.search(r'\d+(?:\.\d+)?', text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _pick_first(row: Dict, keys: List[str]) -> str:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(key.lower())
        if value:
            return str(value).strip()
    return ''


def _load_journal_metrics() -> Dict[str, Dict]:
    global _JOURNAL_METRIC_CACHE
    if _JOURNAL_METRIC_CACHE is not None:
        return _JOURNAL_METRIC_CACHE

    metrics = {
        _normalize_journal_name(name): {
            'impact_factor': BUILTIN_HIGH_IMPACT_JOURNAL_FLOOR,
            'impact_factor_label': '>=10 (builtin high-impact journal list)',
            'impact_factor_source': 'builtin_high_impact_journal_list',
        }
        for name in BUILTIN_HIGH_IMPACT_JOURNALS
    }

    if JOURNAL_METRICS_CSV:
        csv_path = Path(JOURNAL_METRICS_CSV)
        try:
            with csv_path.open(newline='', encoding='utf-8-sig') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    journal = _pick_first(
                        row,
                        ['journal', 'journal_title', 'full_title', 'title', 'source title', 'journal name'],
                    )
                    impact_factor = _parse_float(
                        _pick_first(row, ['impact_factor', 'journal_impact_factor', 'jif', 'if', 'impact factor'])
                    )
                    normalized = _normalize_journal_name(journal)
                    if normalized and impact_factor is not None:
                        metrics[normalized] = {
                            'impact_factor': impact_factor,
                            'impact_factor_label': f'{impact_factor:g}',
                            'impact_factor_source': str(csv_path),
                        }
        except Exception as exc:
            print(f"[PubMed] failed to load journal metrics CSV {csv_path}: {exc}")

    _JOURNAL_METRIC_CACHE = metrics
    return metrics


def _coerce_recent_years(value) -> int:
    if value is None:
        return DEFAULT_RECENT_YEARS
    try:
        years = int(value)
    except (TypeError, ValueError):
        years = DEFAULT_RECENT_YEARS
    return max(1, min(years, 30))


def _coerce_min_impact_factor(value) -> Optional[float]:
    if value is None:
        value = DEFAULT_MIN_IMPACT_FACTOR
    impact_factor = _parse_float(value)
    if impact_factor is None or impact_factor <= 0:
        return None
    return impact_factor


def _clamp_result_count(max_results: int) -> int:
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = 8
    return max(5, min(max_results, 10))


def _extract_year(value: str) -> int:
    match = re.search(r'(19|20)\d{2}', value or '')
    return int(match.group(0)) if match else 0


def _clean_text(value: str) -> str:
    value = html.unescape(value or '')
    return re.sub(r'\s+', ' ', value).strip()


class PubMedSearcher:
    """PubMed文献搜索工具"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or PUBMED_API_KEY
        self.base_url = PUBMED_API_BASE

    def _with_api_key(self, params: Dict) -> Dict:
        if self.api_key:
            params['api_key'] = self.api_key
        return params

    def search(
        self,
        query: str,
        max_results: int = 8,
        sort: str = 'pub date',
        start_year: int = None,
        min_impact_factor: float = None,
        candidate_pool_size: int = None,
    ) -> List[Dict]:
        """
        搜索PubMed文献，并按年份从近到远返回。

        Args:
            query: PubMed检索式
            max_results: 返回5-10篇
            sort: ESearch排序方式，默认按发表日期
        """
        max_results = _clamp_result_count(max_results)
        result_limit = max_results
        if candidate_pool_size is not None:
            try:
                result_limit = max(result_limit, int(candidate_pool_size))
            except (TypeError, ValueError):
                result_limit = max_results
        result_limit = max(1, min(result_limit, 100))
        retmax = min(200, max(result_limit * 3, max_results * 3))
        search_url = f'{self.base_url}/esearch.fcgi'
        params = self._with_api_key({
            'db': 'pubmed',
            'term': query,
            'retmax': retmax,
            'sort': sort,
            'retmode': 'json',
        })

        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            pmids = data.get('esearchresult', {}).get('idlist', [])
            if not pmids:
                return [{'error': '未找到相关文献', 'query': query}]

            results = self._fetch_details(pmids)
            results = self._filter_quality(results, start_year=start_year, min_impact_factor=min_impact_factor)
            results = self._sort_and_limit(results, result_limit)
            if results:
                return results
            return [{'error': self._quality_filter_error(start_year, min_impact_factor), 'query': query}]

        except Exception as e:
            return [{'error': f'搜索失败: {str(e)}', 'query': query}]

    def _fetch_details(self, pmids: List[str]) -> List[Dict]:
        """通过EFetch获取题名、摘要、年份、期刊、DOI等详细信息。"""
        fetch_url = f'{self.base_url}/efetch.fcgi'
        params = self._with_api_key({
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
        })

        try:
            response = requests.get(fetch_url, params=params, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.text)

            results = []
            for article in root.findall('.//PubmedArticle'):
                pmid = self._first_text(article, './/PMID')
                if not pmid:
                    continue

                title = self._join_text(article.find('.//ArticleTitle'))
                journal_title = self._first_text(article, './/Journal/Title')
                journal_medline = self._first_text(article, './/MedlineTA')
                journal_iso = self._first_text(article, './/Journal/ISOAbbreviation')
                journal = journal_title or journal_medline or journal_iso
                journal_metric = self._journal_metric([journal_title, journal_medline, journal_iso])
                date = self._extract_pub_date(article)
                year = _extract_year(date)
                abstract = self._extract_abstract(article)
                authors = self._extract_authors(article)
                doi = self._extract_article_id(article, 'doi')
                publication_types = [
                    _clean_text(''.join(node.itertext()))
                    for node in article.findall('.//PublicationType')
                ]
                mesh_terms = [
                    _clean_text(''.join(node.itertext()))
                    for node in article.findall('.//MeshHeading/DescriptorName')
                ]

                results.append({
                    'pmid': pmid,
                    'title': title or 'N/A',
                    'authors': authors[:5],
                    'journal': journal or 'N/A',
                    'journal_iso': journal_iso,
                    'journal_medline': journal_medline,
                    'date': date or 'N/A',
                    'year': year,
                    'impact_factor': journal_metric.get('impact_factor'),
                    'impact_factor_label': journal_metric.get('impact_factor_label'),
                    'impact_factor_source': journal_metric.get('impact_factor_source'),
                    'abstract': abstract,
                    'doi': doi,
                    'publication_types': publication_types,
                    'mesh_terms': mesh_terms,
                    'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                })

            return results

        except Exception as e:
            return [{'error': f'获取详情失败: {str(e)}'}]

    def _sort_and_limit(self, results: List[Dict], max_results: int) -> List[Dict]:
        valid = [item for item in results if 'error' not in item and item.get('pmid')]
        valid.sort(
            key=lambda item: (
                item.get('year') or 0,
                item.get('impact_factor') or 0.0,
                item.get('pmid') or '',
            ),
            reverse=True,
        )
        return valid[:max_results]

    @staticmethod
    def _journal_metric(journal_names: List[str]) -> Dict:
        metrics = _load_journal_metrics()
        for name in journal_names:
            normalized = _normalize_journal_name(name)
            if normalized and normalized in metrics:
                return metrics[normalized]
            if normalized:
                for metric_name, metric in metrics.items():
                    if normalized.startswith(metric_name + ' ') or metric_name.startswith(normalized + ' '):
                        return metric
        return {}

    @staticmethod
    def _filter_quality(
        results: List[Dict],
        start_year: int = None,
        min_impact_factor: float = None,
    ) -> List[Dict]:
        filtered = []
        for item in results:
            if 'error' in item:
                continue
            year = item.get('year') or _extract_year(item.get('date', ''))
            if start_year and (not year or year < start_year):
                continue
            impact_factor = item.get('impact_factor')
            if min_impact_factor is not None:
                if impact_factor is None or impact_factor < min_impact_factor:
                    continue
            filtered.append(item)
        return filtered

    @staticmethod
    def _quality_filter_error(start_year: int = None, min_impact_factor: float = None) -> str:
        filters = []
        if start_year:
            filters.append(f'publication year >= {start_year}')
        if min_impact_factor is not None:
            filters.append(f'journal impact factor >= {min_impact_factor:g}')
        suffix = '; '.join(filters) if filters else 'current query'
        return f'No PubMed records matched filters: {suffix}'

    @staticmethod
    def _combined_text(item: Dict) -> str:
        parts = [
            item.get('title', ''),
            item.get('abstract', ''),
            ' '.join(item.get('mesh_terms', [])),
            ' '.join(item.get('publication_types', [])),
        ]
        return _clean_text(' '.join(parts)).lower()

    def _matches_target_cancer(self, item: Dict, cancer_type: str) -> bool:
        cancer_key = (cancer_type or '').upper()
        text = self._combined_text(item)
        patterns = CANCER_FILTER_PATTERNS.get(cancer_key)
        if not patterns:
            cancer_text = (cancer_type or '').strip().lower()
            return bool(cancer_text and cancer_text in text)
        return any(re.search(pattern, text, flags=re.I) for pattern in patterns)

    def _is_treatment_relevant(self, item: Dict) -> bool:
        text = self._combined_text(item)
        pub_types = ' '.join(item.get('publication_types', [])).lower()
        has_treatment_signal = any(re.search(pattern, text, flags=re.I) for pattern in TREATMENT_FILTER_PATTERNS)
        has_clinical_type = any(
            term in pub_types
            for term in ['clinical trial', 'guideline', 'meta-analysis', 'systematic review', 'review']
        )
        has_preclinical_signal = any(re.search(pattern, text, flags=re.I) for pattern in PRECLINICAL_PATTERNS)
        has_non_treatment_only_signal = any(
            re.search(pattern, item.get('title', '') + ' ' + item.get('abstract', ''), flags=re.I)
            for pattern in NON_TREATMENT_ONLY_PATTERNS
        )

        if has_preclinical_signal and not has_clinical_type:
            return False
        if has_non_treatment_only_signal and not has_treatment_signal:
            return False
        return has_treatment_signal or has_clinical_type

    def _filter_cancer_treatment_results(self, results: List[Dict], cancer_type: str) -> List[Dict]:
        filtered = []
        for item in results:
            if 'error' in item or not item.get('pmid'):
                continue
            if not self._matches_target_cancer(item, cancer_type):
                continue
            if not self._is_treatment_relevant(item):
                continue
            filtered.append(item)
        return filtered

    @staticmethod
    def _first_text(root: ET.Element, path: str) -> str:
        node = root.find(path)
        return _clean_text(''.join(node.itertext())) if node is not None else ''

    @staticmethod
    def _join_text(node: Optional[ET.Element]) -> str:
        return _clean_text(''.join(node.itertext())) if node is not None else ''

    def _extract_pub_date(self, article: ET.Element) -> str:
        article_date = article.find('.//ArticleDate')
        if article_date is not None:
            year = self._first_text(article_date, 'Year')
            month = self._first_text(article_date, 'Month')
            day = self._first_text(article_date, 'Day')
            return '-'.join(part for part in [year, month, day] if part)

        pub_date = article.find('.//JournalIssue/PubDate')
        if pub_date is None:
            return ''
        year = self._first_text(pub_date, 'Year')
        month = self._first_text(pub_date, 'Month')
        day = self._first_text(pub_date, 'Day')
        medline_date = self._first_text(pub_date, 'MedlineDate')
        return '-'.join(part for part in [year, month, day] if part) or medline_date

    @staticmethod
    def _extract_abstract(article: ET.Element) -> str:
        parts = []
        for node in article.findall('.//Abstract/AbstractText'):
            label = node.attrib.get('Label')
            text = _clean_text(''.join(node.itertext()))
            if text:
                parts.append(f"{label}: {text}" if label else text)
        return _clean_text(' '.join(parts))

    @staticmethod
    def _extract_authors(article: ET.Element) -> List[Dict]:
        authors = []
        for node in article.findall('.//Author'):
            last_name = _clean_text(node.findtext('LastName', ''))
            initials = _clean_text(node.findtext('Initials', ''))
            collective = _clean_text(node.findtext('CollectiveName', ''))
            name = collective or ' '.join(part for part in [last_name, initials] if part)
            if name:
                authors.append({'name': name})
        return authors

    @staticmethod
    def _extract_article_id(article: ET.Element, id_type: str) -> str:
        for node in article.findall('.//ArticleId'):
            if node.attrib.get('IdType') == id_type:
                return _clean_text(node.text or '')
        return ''

    def search_cancer_treatment(
        self,
        cancer_type: str,
        stage: str = None,
        treatment: str = None,
        keywords: str = None,
        max_results: int = 8,
        recent_years: int = None,
        min_impact_factor: float = None,
    ) -> List[Dict]:
        """搜索癌症治疗相关文献，优先返回近年临床治疗证据。"""
        max_results = _clamp_result_count(max_results)
        current_year = datetime.now().year
        recent_years = _coerce_recent_years(recent_years)
        min_impact_factor = _coerce_min_impact_factor(min_impact_factor)
        start_year = current_year - recent_years + 1
        candidate_pool_size = max(40, max_results * 8)
        cancer_key = (cancer_type or '').upper()
        cancer_query = CANCER_QUERY_MAP.get(cancer_key, cancer_type or '')
        base_parts = [
            f"({cancer_query})",
            f"({TREATMENT_TERMS})",
            "(Humans[Mesh] OR human[Title/Abstract])",
            "NOT (animals[Mesh] NOT humans[Mesh])",
        ]
        if stage:
            base_parts.append(f'("{stage}"[Title/Abstract] OR "{stage}"[All Fields])')
        if treatment:
            base_parts.append(f"({treatment})")

        base_query = " AND ".join(base_parts)
        query_variants = [
            f"{base_query} AND (clinical trial[Publication Type] OR randomized[Title/Abstract] OR guideline[Publication Type] OR meta-analysis[Publication Type]) AND ({start_year}:{current_year}[PDat])",
            f"{base_query} AND ({start_year}:{current_year}[PDat])",
            f"{base_query} AND (clinical trial[Publication Type] OR guideline[Publication Type] OR review[Publication Type])",
            base_query,
            f"({cancer_query}) AND ({TREATMENT_TERMS}) AND (clinical trial[Publication Type] OR guideline[Publication Type] OR review[Publication Type])",
        ]
        if keywords:
            query_variants.insert(
                1,
                f"{base_query} AND ({keywords}) AND ({start_year}:{current_year}[PDat])",
            )

        all_results = []
        seen_pmids = set()
        last_error = None
        for query in query_variants:
            results = self.search(
                query,
                max_results=max_results,
                sort='pub date',
                start_year=start_year,
                min_impact_factor=min_impact_factor,
                candidate_pool_size=candidate_pool_size,
            )
            if results and 'error' in results[0]:
                last_error = results[0]
                continue

            filtered_results = self._filter_cancer_treatment_results(results, cancer_key)
            for item in filtered_results:
                pmid = item.get('pmid')
                if pmid and pmid not in seen_pmids:
                    item['query'] = query
                    all_results.append(item)
                    seen_pmids.add(pmid)

            if len(all_results) >= max_results:
                break

        if not all_results:
            return [last_error or {'error': '未找到目标癌种治疗相关文献，已过滤非治疗或非目标癌种结果', 'query': base_query}]

        return self._sort_and_limit(all_results, max_results)


# 全局实例
_searcher = None


def get_pubmed_searcher(api_key: str = None) -> PubMedSearcher:
    """获取PubMed搜索器实例"""
    global _searcher
    if _searcher is None or api_key:
        _searcher = PubMedSearcher(api_key)
    return _searcher


def search_pubmed(query: str, max_results: int = 8) -> List[Dict]:
    """搜索PubMed文献（便捷函数）"""
    searcher = get_pubmed_searcher()
    return searcher.search(query, max_results)


def search_cancer_literature(
    cancer_type: str,
    stage: str = None,
    treatment: str = None,
    keywords: str = None,
    max_results: int = 8,
    recent_years: int = None,
    min_impact_factor: float = None,
) -> List[Dict]:
    """搜索癌症治疗相关文献（便捷函数）"""
    searcher = get_pubmed_searcher()
    return searcher.search_cancer_treatment(
        cancer_type=cancer_type,
        stage=stage,
        treatment=treatment,
        keywords=keywords,
        max_results=max_results,
        recent_years=recent_years,
        min_impact_factor=min_impact_factor,
    )


def format_pubmed_results(results: List[Dict]) -> str:
    """格式化PubMed搜索结果为可引用证据字符串。"""
    if not results:
        return '未找到相关文献'

    if 'error' in results[0]:
        query = results[0].get('query', '')
        return f'搜索出错: {results[0].get("error")}；query={query}'

    output = []
    for i, item in enumerate(results, 1):
        title = item.get('title', 'N/A')
        journal = item.get('journal', 'N/A')
        date = item.get('date', 'N/A')
        year = item.get('year') or _extract_year(date)
        pmid = item.get('pmid', '')
        url = item.get('url', '')
        doi = item.get('doi', '')
        abstract = item.get('abstract', '')
        pub_types = ', '.join(item.get('publication_types', [])[:3])
        impact_factor_label = item.get('impact_factor_label')
        impact_factor_source = item.get('impact_factor_source')
        if not impact_factor_label and item.get('impact_factor') is not None:
            impact_factor_label = f'{item.get("impact_factor"):g}'
        impact_factor_text = impact_factor_label or 'N/A'
        if impact_factor_source:
            impact_factor_text = f'{impact_factor_text}; source={impact_factor_source}'

        authors = item.get('authors', [])
        author_str = authors[0].get('name', 'Unknown') if authors else 'Unknown'
        if len(authors) > 1:
            author_str += ' et al.'

        evidence = abstract[:500] + ('...' if len(abstract) > 500 else '')
        if not evidence:
            evidence = '摘要未提供，请基于题名、期刊和研究类型谨慎引用。'

        output.append(
            f'【PubMed-{i} | PMID: {pmid} | 年份: {year or "N/A"}】\n'
            f'题名: {title}\n'
            f'作者: {author_str}\n'
            f'期刊: {journal}, {date}\n'
            f'影响因子: {impact_factor_text}\n'
            f'研究类型: {pub_types or "N/A"}\n'
            f'治疗相关证据摘要: {evidence}\n'
            f'引用标注: [PMID: {pmid}]\n'
            f'DOI: {doi or "N/A"}\n'
            f'链接: {url}'
        )

    return '\n\n'.join(output)


if __name__ == '__main__':
    print('测试PubMed搜索...')
    results = search_cancer_literature(
        cancer_type='BRCA',
        stage='Stage II',
        treatment='adjuvant therapy',
        max_results=5,
    )
    print(format_pubmed_results(results))
