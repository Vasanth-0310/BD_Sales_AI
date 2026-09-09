"""Document text extraction utilities for PDF and DOCX files."""

import io
from pathlib import PurePosixPath

from src.common.logger import get_logger
from src.domain.exceptions.rag_exceptions import DocumentExtractionError

logger = get_logger(__name__)


class DocumentExtractor:
    """Pure utility class for extracting text content from document files.

    Supports ``.pdf`` and ``.docx`` formats.  Any other extension will raise
    a :class:`DocumentExtractionError`.
    """

    @staticmethod
    def extract_text(file_bytes: bytes, filename: str) -> str:
        """Extract plain text from the raw bytes of a document file.

        Parameters
        ----------
        file_bytes:
            Raw binary content of the uploaded file.
        filename:
            Original filename including extension (e.g. ``"resume.pdf"``).

        Returns
        -------
        str
            The concatenated text content of the document.

        Raises
        ------
        DocumentExtractionError
            If the file type is unsupported or extraction fails for any reason.
        """
        extension = PurePosixPath(filename).suffix.lower()
        logger.info(
            "Starting text extraction for '%s' (extension=%s, size=%d bytes)",
            filename,
            extension,
            len(file_bytes),
        )

        try:
            if extension == ".docx":
                text = DocumentExtractor._extract_docx(file_bytes)
            elif extension == ".pdf":
                text = DocumentExtractor._extract_pdf(file_bytes)
            elif extension == ".txt":
                # Plain-text passthrough — used by scripts/sync_postgres_to_qdrant.py
                # which feeds in-memory DB text under a "<name>.txt" filename.
                text = file_bytes.decode("utf-8", errors="replace")
            else:
                raise DocumentExtractionError(
                    reason=f"Unsupported file extension: '{extension}'"
                )
        except DocumentExtractionError:
            raise
        except Exception as e:
            raise DocumentExtractionError(reason=str(e)) from e

        logger.info(
            "Extraction complete for '%s': %d characters extracted",
            filename,
            len(text),
        )
        return text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        """Extract text from a DOCX file using ``python-docx``.

        Parameters
        ----------
        file_bytes:
            Raw binary content of the ``.docx`` file.

        Returns
        -------
        str
            Paragraph texts joined by newlines.
        """
        from docx import Document  # type: ignore[import-untyped]

        document = Document(io.BytesIO(file_bytes))
        parts = [paragraph.text for paragraph in document.paragraphs]

        # document.paragraphs only covers body paragraphs — ALL table content
        # (tech matrices, metrics, architecture tables) would be silently lost.
        # Iterate every table cell in document order.
        for table in document.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                row_text = " | ".join(cell for cell in row_cells if cell)
                if row_text:
                    parts.append(row_text)

        return "\n".join(parts)

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> str:
        """Extract text from a PDF file using ``pdfplumber``.

        Parameters
        ----------
        file_bytes:
            Raw binary content of the ``.pdf`` file.

        Returns
        -------
        str
            Page texts joined by newlines.
        """
        import pdfplumber  # type: ignore[import-untyped]

        pages_text: list[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
        return "\n".join(pages_text)
