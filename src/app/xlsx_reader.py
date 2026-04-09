"""Small XLSX reader built on top of the standard library."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


SPREADSHEET_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _column_index(cell_reference: str) -> int:
    index = 0
    for character in cell_reference:
        if character.isalpha():
            index = index * 26 + (ord(character.upper()) - 64)
    return max(index - 1, 0)


def _read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared_strings: list[str] = []
    for item in root.findall("main:si", SPREADSHEET_NS):
        shared_strings.append(
            "".join(
                text_node.text or ""
                for text_node in item.iterfind(".//main:t", SPREADSHEET_NS)
            )
        )
    return shared_strings


def _sheet_targets(archive: ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relation_by_id = {
        relation.attrib["Id"]: relation.attrib["Target"]
        for relation in relationships.findall("pkg:Relationship", SPREADSHEET_NS)
    }

    targets: dict[str, str] = {}
    for sheet in workbook.findall("main:sheets/main:sheet", SPREADSHEET_NS):
        relation_id = sheet.attrib.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        )
        if not relation_id:
            continue
        target = relation_by_id.get(relation_id)
        if not target:
            continue
        targets[sheet.attrib["name"]] = f"xl/{target}"
    return targets


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            text_node.text or ""
            for text_node in cell.iterfind(".//main:t", SPREADSHEET_NS)
        )

    value_node = cell.find("main:v", SPREADSHEET_NS)
    if value_node is None or value_node.text is None:
        return ""

    if cell_type == "s":
        return shared_strings[int(value_node.text)]
    if cell_type == "b":
        return "TRUE" if value_node.text == "1" else "FALSE"
    return value_node.text


def read_xlsx_sheet(path: str | Path, sheet_name: str | None = None) -> list[list[str]]:
    """Return the raw rows from an XLSX worksheet."""

    workbook_path = Path(path)
    with ZipFile(workbook_path) as archive:
        shared_strings = _read_shared_strings(archive)
        targets = _sheet_targets(archive)
        if not targets:
            return []

        selected_sheet = sheet_name or next(iter(targets))
        sheet_path = targets[selected_sheet]
        worksheet = ET.fromstring(archive.read(sheet_path))

        rows: list[list[str]] = []
        for row in worksheet.findall("main:sheetData/main:row", SPREADSHEET_NS):
            values_by_index: dict[int, str] = {}
            for cell in row.findall("main:c", SPREADSHEET_NS):
                reference = cell.attrib.get("r", "")
                values_by_index[_column_index(reference)] = _cell_value(
                    cell, shared_strings
                )

            if not values_by_index:
                rows.append([])
                continue

            max_index = max(values_by_index)
            rows.append([values_by_index.get(index, "") for index in range(max_index + 1)])

    return rows


def read_xlsx_records(path: str | Path, sheet_name: str | None = None) -> list[dict[str, str]]:
    """Return worksheet rows mapped by header names."""

    rows = read_xlsx_sheet(path, sheet_name=sheet_name)
    if not rows:
        return []

    headers = rows[0]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        padded_row = row + [""] * max(len(headers) - len(row), 0)
        record = {
            header: padded_row[index] if index < len(padded_row) else ""
            for index, header in enumerate(headers)
            if header
        }
        if any(value for value in record.values()):
            records.append(record)
    return records
