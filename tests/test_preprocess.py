"""Tests for the text preprocessing module."""
from articulate.preprocess import preprocess_for_tts


class TestAbbreviations:
    """Test that period-based abbreviations are replaced."""

    def test_eg_replaced(self):
        assert "for example," in preprocess_for_tts("This is e.g. a test")

    def test_ie_replaced(self):
        assert "that is," in preprocess_for_tts("This is i.e. a test")

    def test_etc_replaced(self):
        result = preprocess_for_tts("apples, oranges, etc. are fruits")
        assert "etcetera" in result
        assert "etc." not in result

    def test_us_periods_removed(self):
        result = preprocess_for_tts("The U.S. government")
        assert "US" in result
        assert "U.S." not in result

    def test_uk_periods_removed(self):
        result = preprocess_for_tts("The U.K. economy")
        assert "UK" in result

    def test_ussr_expanded(self):
        result = preprocess_for_tts("The USSR collapsed")
        assert "the Soviet Union" in result
        assert "USSR" not in result


class TestNumberedLists:
    """Test numbered list → ordinal word conversion."""

    def test_inline_numbers(self):
        result = preprocess_for_tts("There are 1) cats, 2) dogs, 3) birds")
        assert "first," in result
        assert "second," in result
        assert "third," in result

    def test_line_start_numbers(self):
        result = preprocess_for_tts("1. Trade wars\n2. Capital wars\n3. Tech wars")
        assert "First:" in result
        assert "Second:" in result
        assert "Third:" in result


class TestHeaders:
    """Test ALL-CAPS header → title case conversion."""

    def test_all_caps_to_title_case(self):
        result = preprocess_for_tts("THE HOT WAR BEGINS")
        assert "The Hot War Begins" in result
        assert "THE HOT WAR" not in result

    def test_header_gets_period(self):
        result = preprocess_for_tts("ECONOMIC WARFARE")
        assert "Economic Warfare." in result


class TestCleanup:
    """Test whitespace and formatting cleanup."""

    def test_double_spaces_removed(self):
        result = preprocess_for_tts("too  many   spaces")
        assert "  " not in result

    def test_triple_newlines_collapsed(self):
        result = preprocess_for_tts("line1\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_escaped_quotes_fixed(self):
        result = preprocess_for_tts('He said \\"hello\\"')
        assert '\\"' not in result
        assert '"' in result


class TestPreservation:
    """Test that normal text is not mangled."""

    def test_normal_sentences_unchanged(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert preprocess_for_tts(text) == text

    def test_regular_periods_preserved(self):
        text = "This is a sentence. This is another."
        assert preprocess_for_tts(text) == text

    def test_numbers_in_text_preserved(self):
        text = "The year was 1941 and prices rose 50 percent."
        result = preprocess_for_tts(text)
        assert "1941" in result
        assert "50 percent" in result
