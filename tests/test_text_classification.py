from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from split_and_categorize import _classify_text_document


def assert_category(name: str, text: str, expected: str) -> None:
    category, confidence = _classify_text_document(text)
    assert category == expected, f"{name}: expected {expected}, got {category} ({confidence})"


def test_cnic_front_outweighs_passport_keywords() -> None:
    assert_category(
        "cnic_front",
        """
        PAKISTAN National Identity Card
        Name Muhammad Zubair
        Father Name Yaseen
        Gender M
        Country of Stay Pakistan
        Identity Number 34102-2165110-9
        Date of Birth 19.09.2002
        Date of Issue 15.03.2021
        Date of Expiry 15.03.2031
        """,
        "cnic",
    )


def test_cnic_back_with_registrar_text() -> None:
    assert_category(
        "cnic_back",
        "34102-2165110-9 Registrar General of Pakistan",
        "cnic",
    )


def test_modern_single_page_cv_headings() -> None:
    assert_category(
        "cv_page",
        """
        MUHAMMAD ZUBAIR
        PROFESSIONAL DRIVER
        CONTACT INFO
        PROFILE
        I am a responsible and hardworking driver with experience in Mazda truck and trailer driving.
        EXPERIENCE
        EDUCATION
        SKILL
        """,
        "cv_resume",
    )