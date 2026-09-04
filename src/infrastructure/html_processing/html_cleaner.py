import json
import re
from bs4 import BeautifulSoup, Comment
from src.common.logger import get_logger

logger = get_logger(__name__)

# Tags to remove entirely from the DOM before text extraction
_TAGS_TO_REMOVE = [
    "script", "style", "noscript", "nav", "footer",
    "iframe", "svg", "img", "figure", "form",
    "button", "input", "select", "textarea", "link", "meta",
]


class HTMLCleaner:
    """
    Cleans raw rendered HTML from the browser before sending to the LLM.
    Reduces token usage while preserving all meaningful job-related content.
    """

    def clean(self, raw_html: str) -> str:
        """
        Parse and clean raw HTML, returning plain text of the job content.

        Args:
            raw_html: Full rendered HTML string from the browser.

        Returns:
            A clean, normalized plain-text string ready for LLM extraction.
        """
        soup = BeautifulSoup(raw_html, "lxml")

        # Step 0: Extract only real JobPosting structured data.
        structured_job_text = self._extract_structured_job_text(soup)

        # Step 1: Extract alt text from images before removing them
        for img in soup.find_all("img"):
            alt_text = img.get("alt")
            if alt_text and alt_text.strip():
                img.replace_with(f" [Image: {alt_text.strip()}] ")

        # Step 2: Remove all irrelevant tags
        for tag in _TAGS_TO_REMOVE:
            for element in soup.find_all(tag):
                element.decompose()

        # Step 2: Remove HTML comments.
        # F2: BeautifulSoup parses comments into `Comment` objects whose
        # string value is the comment CONTENT (no `<!--` delimiters), so a
        # startswith("<!--") string check never matched. isinstance() does.
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Step 3: Fall back to full body if no container found
        target = soup.body or soup

        # Step 4: Extract and normalize text
        raw_text = target.get_text(separator="\n")
        
        if structured_job_text:
            raw_text = structured_job_text + "\n\n--- PAGE TEXT ---\n" + raw_text

        cleaned = self._normalize_whitespace(raw_text)

        logger.debug(f"HTML cleaned: {len(raw_html)} bytes -> {len(cleaned)} chars")
        return cleaned

    def _extract_structured_job_text(self, soup: BeautifulSoup) -> str:
        """Extract compact schema.org JobPosting data from JSON-LD scripts only."""
        sections: list[str] = []

        for script in soup.find_all("script", type="application/ld+json"):
            if not script.string:
                continue
            try:
                data = json.loads(script.string)
            except json.JSONDecodeError:
                continue

            for posting in self._find_job_postings(data):
                lines = self._format_job_posting(posting)
                if lines:
                    sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _find_job_postings(self, value):
        if isinstance(value, dict):
            raw_type = value.get("@type")
            types = raw_type if isinstance(raw_type, list) else [raw_type]
            if any(str(item).lower() == "jobposting" for item in types):
                yield value

            graph = value.get("@graph")
            if graph:
                yield from self._find_job_postings(graph)

        elif isinstance(value, list):
            for item in value:
                yield from self._find_job_postings(item)

    def _format_job_posting(self, posting: dict) -> list[str]:
        lines: list[str] = []

        def add(label: str, value) -> None:
            text = self._structured_value_to_text(value)
            if text:
                lines.append(f"{label}: {text}")

        add("Title", posting.get("title"))
        add("Company", posting.get("hiringOrganization"))
        add("Location", posting.get("jobLocation"))
        add("Employment type", posting.get("employmentType"))
        add("Date posted", posting.get("datePosted"))
        add("Valid through", posting.get("validThrough"))
        add("Salary", posting.get("baseSalary"))
        add("Description", posting.get("description"))

        return lines

    def _structured_value_to_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return self._html_to_text(value)
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return "; ".join(
                text for text in (self._structured_value_to_text(item) for item in value) if text
            )
        if isinstance(value, dict):
            preferred = []
            for key in (
                "name",
                "address",
                "streetAddress",
                "addressLocality",
                "addressRegion",
                "postalCode",
                "addressCountry",
                "value",
                "minValue",
                "maxValue",
                "unitText",
                "currency",
            ):
                if key in value:
                    text = self._structured_value_to_text(value.get(key))
                    if text:
                        preferred.append(text)
            if preferred:
                return ", ".join(preferred)
        return ""

    @staticmethod
    def _html_to_text(value: str) -> str:
        return BeautifulSoup(value, "lxml").get_text(separator="\n", strip=True)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse multiple blank lines and strip leading/trailing whitespace."""
        # Replace tabs with space
        text = text.replace("\t", " ")
        # Collapse 3+ consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in text.splitlines()]
        # Remove fully empty leading/trailing lines
        text = "\n".join(lines).strip()
        return text
