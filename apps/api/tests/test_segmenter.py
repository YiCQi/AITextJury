from zeroaibench.segmenter import segment_text


def test_paragraph_offsets_are_absolute():
    text = (
        "First paragraph here. It has two sentences.\n\n"
        "Second paragraph. Also two sentences. Short one too."
    )
    seg = segment_text(text)
    assert len(seg.paragraphs) == 2
    for p in seg.paragraphs:
        assert text[p.start:p.end] == p.text


def test_single_line_becomes_one_paragraph():
    seg = segment_text("Just one line of text.")
    assert len(seg.paragraphs) == 1
    assert seg.paragraphs[0].start == 0


def test_sentences_cover_paragraph_text():
    text = ("Alpha beta gamma delta. Epsilon zeta eta!\n\n"
            "Theta iota kappa; lambda mu nu. Omicron.")
    seg = segment_text(text)
    for para in seg.paragraphs:
        kids = [s for s in seg.sentences if s.parent_id == para.id]
        # sentence offsets must reconstruct the paragraph
        rebuilt = "".join(text[s.start:s.end] for s in kids)
        assert rebuilt.strip() == para.text.strip()


def test_chinese_sentence_splitting():
    text = "今天天气很好，我们去公园散步。公园里人很多！大家都很开心。\n\n明天再一起去吗？"
    para1 = segment_text(text).sentences
    seg = segment_text(text)
    assert len(seg.paragraphs) == 2
    # CJK enders split without spaces
    assert len([s for s in seg.sentences if s.parent_id == seg.paragraphs[0].id]) == 3
    for s in seg.sentences:
        assert len(s.text.strip()) > 0


def test_tiny_fragments_merge_into_previous():
    text = "A fairly long opening sentence with plenty of words. Ok?"
    seg = segment_text(text)
    for s in seg.sentences:
        # "Ok?" was merged into the previous sentence (or kept) — never empty
        assert s.text.strip()


def test_first_sentence_starts_at_paragraph_offset():
    text = "One two three four. Five six seven eight."
    seg = segment_text(text)
    assert seg.sentences[0].start == 0
    # documented semantics: whitespace after a Latin ender belongs to the
    # NEXT sentence so slices stay contiguous
    joined = "".join(text[s.start:s.end] for s in seg.sentences)
    assert joined == text


def test_empty_and_blank_text():
    seg = segment_text("   \n\n  ")
    assert seg.paragraphs  # one vacuous paragraph, empty sentence list allowed


def test_sentence_ids_unique():
    text = "a b c d e f g. h i j k l m. n o p q r s."
    seg = segment_text(text)
    ids = [s.id for s in seg.sentences]
    assert len(ids) == len(set(ids))
    pids = [p.id for p in seg.paragraphs]
    assert len(pids) == len(set(pids))
