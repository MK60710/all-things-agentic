from __future__ import annotations

from agent.bibliography import build_bibtex


def test_build_bibtex_returns_a_placeholder_comment_for_no_papers():
    assert build_bibtex([]) == "% No papers in this session yet.\n"


def test_build_bibtex_arxiv_paper_gets_year_eprint_archiveprefix_and_url():
    papers = [
        {
            "id": "arxiv-2106.09685",
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "authors": "Edward J. Hu, Yelong Shen, Weizhu Chen",
            "abstract": "We propose LoRA.",
        }
    ]

    bibtex = build_bibtex(papers)

    assert "@misc{arxiv_2106_09685," in bibtex
    assert "title = {LoRA: Low-Rank Adaptation of Large Language Models}" in bibtex
    assert "author = {Edward J. Hu and Yelong Shen and Weizhu Chen}" in bibtex
    assert "year = {2021}" in bibtex
    assert "eprint = {2106.09685}" in bibtex
    assert "archivePrefix = {arXiv}" in bibtex
    assert "url = {https://arxiv.org/abs/2106.09685}" in bibtex


def test_build_bibtex_uploaded_paper_omits_arxiv_only_fields():
    papers = [{"id": "my-uploaded-paper", "title": "A Local PDF", "authors": None}]

    bibtex = build_bibtex(papers)

    assert "@misc{my_uploaded_paper," in bibtex
    assert "title = {A Local PDF}" in bibtex
    assert "year" not in bibtex
    assert "eprint" not in bibtex
    assert "archivePrefix" not in bibtex
    assert "author" not in bibtex


def test_build_bibtex_prefers_a_real_pdf_url_over_the_derived_arxiv_link():
    papers = [
        {
            "id": "arxiv-2106.09685",
            "title": "LoRA",
            "pdf_url": "https://arxiv.org/pdf/2106.09685v2",
        }
    ]

    bibtex = build_bibtex(papers)

    assert "url = {https://arxiv.org/pdf/2106.09685v2}" in bibtex


def test_build_bibtex_multiple_papers_produces_one_entry_each():
    papers = [
        {"id": "arxiv-1706.03762", "title": "Attention Is All You Need"},
        {"id": "arxiv-2106.09685", "title": "LoRA"},
    ]

    bibtex = build_bibtex(papers)

    assert bibtex.count("@misc{") == 2
    assert "arxiv_1706_03762" in bibtex
    assert "arxiv_2106_09685" in bibtex


def test_build_bibtex_escapes_stray_braces_in_free_text_fields():
    papers = [{"id": "paper-a", "title": "A {weird} title"}]

    bibtex = build_bibtex(papers)

    assert "title = {A weird title}" in bibtex
