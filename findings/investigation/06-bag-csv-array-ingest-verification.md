# Bag CSV → SQLite array-column ingest verification

**Investigator:** ArrayAsJson bag-CSV path verification (research-only)
**Date:** 2026-05-28
**Scope:** deriva-py @ HEAD on e2e-test/2026-05-28 worktree
(`/Users/carl/GitHub/deriva-py`, the PR #265 ArrayAsJson fix already
landed). Live reproduction in a `:memory:` SQLite via the actual
`deriva.bag._column_types.ArrayAsJson` class.
**Mode:** No code changes. Closes the gap explicitly flagged by
finding 04 §5 ("bag-mode write path — not reproduced live").

---

## 1. TL;DR

**Bug confirmed.** PR #265 fixed Path A (live-catalog Python `list`
inputs) but did NOT fix Path B (bag-CSV PG-literal-string inputs).
For Path B the SQLite cell holds a JSON-encoded **string** of the PG
literal (`'"{plane,aeroplane}"'`), not a JSON-encoded list
(`'["plane","aeroplane"]'`). The `ArrayAsJson` docstring's
end-to-end-`list` round-trip claim does not hold for the CSV input
path.

The bug is **observable today** through any consumer of the bag's
SQLite mirror that reads array columns — including
`DatasetBag.get_denormalized_as_dataframe` for any include set that
crosses a vocab table. It is **hidden on the catalog write-back path
only** by `BagCatalogLoader._coerce_pg_array`
(`catalog_loader.py:1344`), which re-parses the PG literal on the way
out. That safety net is the reason this hasn't visibly broken bag
clones; it doesn't help any other reader of the SQLite mirror.

**Recommended fix:** override `ArrayAsJson.process_bind_param` in
`/Users/carl/GitHub/deriva-py/deriva/bag/_column_types.py`. Detect
PG-literal strings on bind, decode to Python `list`, then delegate
to `JSON.bind_processor` (the existing flow). Single file, single
class, fully type-system-local. **Do not** add row-walking
pre-coercion in `DataLoader._insert_batch` — it duplicates the
schema walk `catalog_loader.py:1335-1339` already does and leaks
type concerns into the loader.

After the fix, **`BagCatalogLoader._coerce_pg_array` becomes dead
code on the bag→catalog write-back path**: `ArrayAsJson` will return
a Python `list` on result-row read, the `not isinstance(value, str)`
early return at `catalog_loader.py:1220` short-circuits the function
to a no-op, and the per-column walk at
`catalog_loader.py:1335-1339` could be deleted entirely. Keep the
function itself if any other caller is added later, but the array-
column branch in `_insert_rows` is the only existing call site.

Broken docstring: `ArrayAsJson` at `_column_types.py:162-180`. The
claim "JSON-encode on write, decode on read, so callers see ``list``
end-to-end" is true for Python-list inputs only. The CSV-string path
violates it silently.

---

## 2. Reproduction output (verbatim)

Live session, deriva-py worktree at
`/Users/carl/GitHub/deriva-py` (PR #265 ArrayAsJson already
present), Python 3.13 via `uv run`.

```
============================================================
Path A: live deriva-ml input (Python list)
============================================================
Path A typed result: [('A', ['plane', 'aeroplane'])]
Path A raw SQLite cell: ('["plane", "aeroplane"]',)

============================================================
Path B: bag CSV input (PG literal string)
============================================================
Path B typed result: [('B', '{plane,aeroplane}')]
Path B raw SQLite cell: ('"{plane,aeroplane}"',)

============================================================
Path C: empty array PG literal
============================================================
Path C typed result: [('C', '{}')]
Path C raw SQLite cell: ('"{}"',)

============================================================
Path D: None
============================================================
Path D typed result: [('D', None)]
Path D raw SQLite cell: ('null',)

============================================================
Path E: PG literal with quoted elements
============================================================
Path E typed result: [('E', '{"a","b"}')]
Path E raw SQLite cell: ('"{\\"a\\",\\"b\\"}"',)

============================================================
Path F: empty string (bag NULL convention)
============================================================
Path F typed result: [('F', '')]
Path F raw SQLite cell: ('""',)

============================================================
Direct json.dumps verification
============================================================
json.dumps('{a,b}') = '"{a,b}"'
json.dumps(['a','b']) = '["a", "b"]'
json.dumps('') = '""'
```

Read these together:

- **Path A vs Path B SQLite cell.** Path A holds `["plane",
  "aeroplane"]` — a JSON array. Path B holds `"{plane,aeroplane}"`
  — a JSON-encoded *string* of the PG literal. Different shapes,
  different round-trip semantics.
- **Path B typed result.** Reading the cell back via SQLAlchemy
  gives the string `"{plane,aeroplane}"`, not the list
  `["plane","aeroplane"]`. Caller observes a `str` where the
  docstring promised a `list`.
- **Path C.** Empty PG literal `"{}"` round-trips as `"{}"` instead
  of `[]`. The empty-array case is also broken.
- **Path D.** `None` round-trips correctly. `JSON`'s NULL handling
  fires before the bind processor, so this case is safe — matches
  the `'null'` storage that SQLAlchemy `JSON` uses.
- **Path E.** Quoted PG literal arrives JSON-encoded with escaped
  quotes (`'"{\\"a\\",\\"b\\"}"'`). Round-trips as the same string,
  no list parsing.
- **Path F.** Empty string round-trips as the empty string `""`.
  Bag CSVs use the empty string as NULL — this is silently the
  *wrong* shape relative to a coerce-to-None contract on a
  nullable array column.

The reproduction is at full fidelity for the static case (the
`csv.DictReader` step does nothing more than yield `str` values; the
`SqliteWriter.write_rows` step does nothing more than
`conn.execute(stmt, rows)` against the same `ArrayAsJson` type the
test instantiates). Live bag execution would produce the same SQLite
cells as Path B / C / E / F.

---

## 3. Code trace — bag CSV input case

The value travels through five frames:

### Frame 1: CSV row read

`/Users/carl/GitHub/deriva-py/deriva/bag/sources.py:175-178` —
`CsvDataSource.get_table_data`:

```python
with csv_file.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        yield row
```

`csv.DictReader` yields `dict[str, str]`. The PG literal column
arrives in the row dict as e.g. `"synonyms": "{plane,aeroplane}"`.
No coercion happens here.

### Frame 2: Loader batches

`/Users/carl/GitHub/deriva-py/deriva/bag/loader.py:847-853` —
`DataLoader._load_table`:

```python
for row in self.source.get_table_data(table):
    batch.append(row)
    if len(batch) >= batch_size:
        rows_loaded += self.sink.write_rows(table, batch)
        batch = []
if batch:
    rows_loaded += self.sink.write_rows(table, batch)
```

Rows pass through untouched. No coercion happens here.

### Frame 3: SQLite sink

`/Users/carl/GitHub/deriva-py/deriva/bag/loader.py:523-565` —
`SqliteWriter.write_rows` ends with:

```python
with self.orm.engine.begin() as conn:
    conn.execute(stmt, rows)
```

The dicts go straight to SQLAlchemy. No coercion happens here.

### Frame 4: `ArrayAsJson` bind

`/Users/carl/GitHub/deriva-py/deriva/bag/_column_types.py:162-180`
— `ArrayAsJson(TypeDecorator)`. The class declares `impl = JSON` and
**no `process_bind_param` override**. The base
`TypeDecorator.bind_processor` delegates to `JSON.bind_processor`.

### Frame 5: SQLAlchemy `JSON.bind_processor`

`/Users/carl/GitHub/deriva-py/.venv/lib/python3.13/site-packages/sqlalchemy/sql/sqltypes.py:2778-2810`
— `JSON._make_bind_processor`:

```python
def process(value):
    if value is self.NULL:
        value = None
    elif isinstance(value, elements.Null) or (
        value is None and self.none_as_null
    ):
        return None
    serialized = json_serializer(value)
    return string_process(serialized)
```

`json_serializer` defaults to `json.dumps`. The value
`"{plane,aeroplane}"` is a `str`, not `self.NULL` and not `None`, so
control reaches `json.dumps("{plane,aeroplane}")` →
`'"{plane,aeroplane}"'` — a 19-character JSON string literal. That
string is what lands in the SQLite cell, verbatim. Confirmed by Path
B's raw cell output.

### Frame 6: Read-back

`JSON.result_processor` runs `json.loads` on the cell. For Path B:
`json.loads('"{plane,aeroplane}"')` → `"{plane,aeroplane}"` (the
PG-literal string). The reader sees a `str`, not a `list`.

This is the divergence from `ArrayAsJson`'s docstring promise.

---

## 4. Code trace — live deriva-ml Python-list case

For comparison, Path A:

- A live HTTP response comes in via `deriva.core` and is
  deserialised by Python's standard JSON parser into a `list`.
- The denormalizer's paged fetcher hands the dict to
  `conn.execute(stmt, rows)`.
- `ArrayAsJson` → `JSON.bind_processor` runs
  `json.dumps(["plane", "aeroplane"])` → `'["plane", "aeroplane"]'`.
- SQLite stores `["plane", "aeroplane"]` as the cell content.
- On read, `json.loads('["plane", "aeroplane"]')` → `["plane",
  "aeroplane"]`. Caller sees `list`.

The end-to-end-`list` contract holds. PR #265 fixed this case (which
is the bind-error case the audit found live). It did not address
Path B.

---

## 5. Docstring vs code analysis

`/Users/carl/GitHub/deriva-py/deriva/bag/_column_types.py:162-180`:

```python
class ArrayAsJson(TypeDecorator):
    """Serialise Python list values as JSON-encoded TEXT for SQLite.

    ERMrest emits array columns (``text[]``, ``int4[]``, ...) as JSON
    arrays, which deriva-py deserialises into Python lists. SQLite has
    no native array type and SQLAlchemy's SQLite dialect cannot bind a
    list to a TEXT column. JSON-encode on write, decode on read, so
    callers see ``list`` end-to-end.

    The wrapped :class:`sqlalchemy.JSON` already round-trips Python
    ``list`` / ``dict`` values transparently on SQLite; the decorator
    exists so the bag's lossless schema round-trip
    (:data:`deriva.bag.schema_io.SQL_TO_ERMREST`) has a named class to
    distinguish *array* columns from scalar ``json`` / ``jsonb``
    columns when no ``ermrest_typename`` is stashed in ``col.info``.
    """

    impl = JSON
    cache_ok = True
```

Claim-by-claim:

| Docstring claim | Reality | Verdict |
|---|---|---|
| "Serialise Python list values as JSON-encoded TEXT for SQLite." | True for `list` inputs (Path A). Untrue for `str` inputs (Path B). | Partial. |
| "ERMrest emits array columns as JSON arrays, which deriva-py deserialises into Python lists." | True for the live-catalog HTTP path. The bag-write CSV path writes PG-literal strings (`{a,b}`); deriva-py does **not** deserialise these to lists when reading bag CSVs. | Misleading by omission. |
| "SQLite has no native array type and SQLAlchemy's SQLite dialect cannot bind a list to a TEXT column." | True. | OK. |
| "JSON-encode on write, decode on read, so callers see ``list`` end-to-end." | False for the bag-CSV input path. `json.dumps("{a,b}") → '"{a,b}"'`; read-back yields the PG-literal string, not the list. | **Broken.** |
| "The wrapped ``sqlalchemy.JSON`` already round-trips Python ``list`` / ``dict`` values transparently on SQLite" | True for Python `list` / `dict` inputs only. The wrapping JSON has no logic for PG-literal-string inputs because that's not what `sqlalchemy.JSON` is for. | Technically OK; the docstring's omission is that it never says what happens for non-`list`/`dict` inputs. |

The single sentence "JSON-encode on write, decode on read, so
callers see ``list`` end-to-end" is the load-bearing claim and the
specific claim Path B refutes. The rest of the docstring is salvable
with a "for ``list`` / ``dict`` inputs" qualifier.

---

## 6. Downstream consumer impact

`grep -rn "synonyms\|Synonyms" /Users/carl/GitHub/deriva-py/deriva/bag/`
shows the bag module itself does **not** read `Synonyms` columns
directly (the two hits are docstring text in `catalog_loader.py`).

But the bag's SQLite mirror is read by external surface area:

1. **`BagDatabase.get_table_contents`**
   (`deriva/bag/database.py:370-374`) — yields raw rows from any
   table in the SQLite mirror. `BagCatalogLoader._load_table` calls
   this at `catalog_loader.py:615-618`. **Hit by the bug**, masked
   by `_coerce_pg_array` further down (see §7).

2. **`DatasetBag.get_denormalized_as_dataframe`** in deriva-ml.
   Issues SQL `SELECT … FROM …` JOINs against the bag's SQLite
   mirror. Any include set that crosses a vocab table will produce a
   dataframe whose `*.Synonyms` column carries PG-literal strings
   (`"{a,b}"`), not `list[str]`. **Observable, unfixed**, no safety
   net.

3. **Other readers** — any user who opens `BagDatabase` directly to
   run their own queries gets the same PG-literal strings.

`_coerce_pg_array` callers:

```
$ grep -rn "_coerce_pg_array" /Users/carl/GitHub/deriva-py/
deriva/bag/catalog_loader.py:1208:    def _coerce_pg_array(value: Any) -> Any:
deriva/bag/catalog_loader.py:1344:                        row[col] = self._coerce_pg_array(row[col])
tests/deriva/bag/test_catalog_loader.py:864-899:        (5 unit tests)
```

**Only one call site in the production code path.** The audit's
"only one caller" assertion is verified.

deriva-ml side: `grep -n "Synonyms\|synonyms"` in
`/Users/carl/GitHub/DerivaML/deriva-ml/src/` shows the live-catalog
`ermrest.py` model uses Synonyms (where the wire format is JSON,
not PG literal, so unaffected). The local-db denormalizer does
not directly reference `Synonyms`, but a user-issued
`include_tables=["Image", "Image_Class"]` against a `DatasetBag`
runs through SQL joins that will surface whatever the
`Synonyms` column holds — and on the bag side that's a PG-literal
string today.

So: **yes, this bug is observable by a real consumer
(`DatasetBag.get_denormalized_as_dataframe`)**, but the most
common consumer (the catalog write-back path in
`BagCatalogLoader`) is shielded by `_coerce_pg_array`. The
audit's "latent" framing was wrong in detail — the bug isn't
latent, it's invisible to the path that has tests and visible to
the path that doesn't.

---

## 7. `_coerce_pg_array` safety-net analysis

`catalog_loader.py:1207-1230`:

```python
@staticmethod
def _coerce_pg_array(value: Any) -> Any:
    """Convert a PostgreSQL CSV array literal into a JSON array."""
    if value is None or not isinstance(value, str):
        return value
    if not (value.startswith("{") and value.endswith("}")):
        return value
    inner = value[1:-1]
    if not inner:
        return []
    return [part.strip().strip('"') for part in inner.split(",")]
```

`catalog_loader.py:1333-1344`:

```python
# Coerce array-typed columns from PostgreSQL literal form
# (``{a,b}``) into real JSON arrays for the wire.
array_cols = [
    c.name
    for c in table.column_definitions
    if getattr(c.type, "is_array", False)
]
if array_cols:
    for row in rows:
        for col in array_cols:
            if col in row:
                row[col] = self._coerce_pg_array(row[col])
```

Verified live with the actual `ArrayAsJson` round-trip in §2:

```
Read-back via SQLAlchemy (result_processor): [('B', '{plane,aeroplane}')]
Type of syn: <class 'str'>
_coerce_pg_array input: '{plane,aeroplane}'
_coerce_pg_array output: ['plane', 'aeroplane']
```

So the bag write-back chain is:

```
CSV "{plane,aeroplane}"
  → SqliteWriter.write_rows
  → ArrayAsJson(JSON).bind_processor: json.dumps("{plane,aeroplane}")
  → SQLite cell holds '"{plane,aeroplane}"'
  → BagDatabase.get_table_contents reads cell
  → ArrayAsJson(JSON).result_processor: json.loads('"{plane,aeroplane}"')
  → "{plane,aeroplane}" (str)
  → _coerce_pg_array(str) → ['plane', 'aeroplane'] (list)
  → ERMrest wire format
```

The safety net is **exactly** `_coerce_pg_array`. Without it, every
bag→catalog clone of any catalog with populated vocab tables would
ship PG-literal strings as the `Synonyms` payload, and ERMrest would
400 with "cannot call json_array_elements_text on a scalar". The
safety net is real and load-bearing today.

After fixing `ArrayAsJson.process_bind_param` to decode PG literals
on bind:

```
CSV "{plane,aeroplane}"
  → ArrayAsJson.process_bind_param: decode PG literal → ['plane', 'aeroplane']
  → JSON.bind_processor: json.dumps([…]) → '["plane", "aeroplane"]'
  → SQLite cell holds '["plane", "aeroplane"]'
  → result_processor: json.loads(…) → ['plane', 'aeroplane'] (list)
  → _coerce_pg_array(list) → list (early return at line 1220, no-op)
  → ERMrest wire format
```

`_coerce_pg_array` becomes a no-op. The `not isinstance(value, str)`
early return short-circuits it, so the function is dead code on this
path. The cleanup follow-up is straightforward: delete the
`array_cols` walk at `catalog_loader.py:1333-1344` and the
`_coerce_pg_array` staticmethod itself. (Keep the test file rows for
the function's intent in case the function is re-added later, or
delete them too — the test corpus catches no regression once the
behavior is upstream.)

---

## 8. Recommended fix location

**Option A: Override `ArrayAsJson.process_bind_param` in
`_column_types.py`.** Recommended.

```python
class ArrayAsJson(TypeDecorator):
    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        # CSV bag-load path: ERMrest's PG-literal array form
        # ("{a,b}" / "{}") arrives as a str. Decode to a list so the
        # wrapped JSON serialises a real array (not a JSON-encoded
        # PG-literal string). The empty string (bag CSV NULL
        # convention) becomes None.
        if isinstance(value, str):
            if value == "":
                return None
            if value.startswith("{") and value.endswith("}"):
                inner = value[1:-1]
                if not inner:
                    return []
                return [part.strip().strip('"') for part in inner.split(",")]
        return value
```

This is the same parsing logic `_coerce_pg_array` already
implements; lift it into the type and the catalog_loader's walk
disappears.

**Option B: Pre-coerce in `DataLoader._insert_batch`** (`loader.py`).
Walk the schema for `is_array` columns and decode PG literals on
each row before bind.

Evaluation:

| Criterion | Option A (process_bind_param) | Option B (loader pre-coerce) |
|---|---|---|
| Locality | One class. | Two responsibilities: loader knows about array types AND owns the decoder. |
| Testability | Direct: feed a string and a list through `process_bind_param`. | Indirect: build a schema with array columns and run the loader. |
| Coverage | Every `ArrayAsJson`-typed write everywhere — bag CSV, hand-written test, any other producer. | Bag CSV only. Misses any direct SQLite writes. |
| Symmetry with read path | `result_processor` already lives in `JSON`. `process_bind_param` on the same class keeps both sides together. | Asymmetric: write coerces, read doesn't (read needs no coercion if write was done right; but two-call-site fix has more surface to mismatch). |
| Defensiveness | `isinstance(value, str)` short-circuit makes the path no-op for list inputs (Path A still works). | Loader-side walk runs for every row even when values are already lists. |
| Interaction with `_coerce_pg_array` | After fix: dead. Clean follow-up. | After fix: still runs because the SQLite read still emits strings — but now redundant with the loader-side coercion. |
| Discoverability | A reader looking at the type sees the round-trip contract. | A reader looking at the type still has to grep the loader. |

**Option A wins.** Recommended fix site:
`/Users/carl/GitHub/deriva-py/deriva/bag/_column_types.py:162-180`,
add a `process_bind_param` method to `ArrayAsJson`.

The companion test belongs at
`/Users/carl/GitHub/deriva-py/tests/deriva/bag/test_column_types.py`
next to the existing `test_array_as_json_round_trips_python_lists_through_sqlite`
— add `test_array_as_json_round_trips_pg_literal_strings_through_sqlite`
covering Path B / C / E / F. The existing Path A test stays valid
unchanged.

---

## 9. Risk assessment

Moving the fix to `ArrayAsJson.process_bind_param`:

| Risk | Severity | Mitigation |
|---|---|---|
| **List input regression.** A caller passes a `list` already; the override should leave it alone. | Low. The `isinstance(value, str)` guard short-circuits non-str inputs. | Test covers Path A and Path B together. |
| **Non-PG-literal strings.** A caller passes a `str` that isn't a PG literal (e.g., a JSON array string `"[\"a\",\"b\"]"`). | Low. The `startswith("{") and endswith("}")` guard means non-PG strings pass through to `JSON` unchanged. That's a regression only if a current user is feeding JSON-array strings into an array column today — which contradicts the type's contract. | Documented in docstring; tested for negative case. |
| **Empty-string-as-NULL behavior change.** Today, a bag CSV empty string in an array column round-trips as `""`. The fix converts it to `None`. | Low. This matches the bag's documented "CSVs serialize NULL as empty string" convention (`_coerce_empty_to_null` at `catalog_loader.py:1232`). | Test that empty string yields None. |
| **PG-literal parsing edge cases.** Embedded commas, quoted commas, NULL elements. | Low. `_coerce_pg_array`'s comment at `catalog_loader.py:1227-1229` already documents: "Naive split is fine for the common case (text[] of simple identifiers, int[] of digits). Embedded commas in quoted strings aren't produced by the current bag walker." Same constraint, same edge case — moving the code doesn't widen the gap. | Identical parser; same edge cases. |
| **Interaction with `_coerce_pg_array`.** Two coercion sites if both run. | None. After the fix, the read-side value is already a list, so the `not isinstance(value, str)` early return makes `_coerce_pg_array` a no-op. The loader-side walk still runs but does nothing. | Remove the walk in a follow-up cleanup. |
| **Cross-version mismatch.** `cache_ok = True` makes SQLAlchemy cache the bind processor. Adding `process_bind_param` invalidates that cache; old code paths might hit stale compiled statements. | Low. Same release; consumers update via lockfile bump. | Standard release flow. |
| **Tests today don't catch the bug.** The bag-CSV-string case has zero unit-test coverage. | Already noted in §10; the fix adds the missing test. | Add Path B / C / E / F to the test corpus. |

Overall: **low risk**. The fix is type-system-local, single-class,
backward-compatible for the input shapes currently in use (Path A
still works), and the parsing logic is the same logic that's
already shipped in `_coerce_pg_array` with the same edge-case
profile.

---

## 10. Open questions

1. **`_coerce_empty_to_null` interaction.** Today,
   `BagCatalogLoader._coerce_empty_to_null` runs before
   `_coerce_pg_array` and converts empty strings to `None` on
   nullable columns. If the fix lands in `process_bind_param`, that
   coercion still runs on the *catalog write-back* but won't run on
   *direct bag-mirror reads*. Open question: do any direct readers
   care about empty-string vs None on array columns?
   `DatasetBag.get_denormalized_as_dataframe` likely passes the value
   straight through to pandas, which treats empty strings and None
   differently in `isna()`. The fix makes the bag-mirror behavior
   match the catalog-write-back behavior, which is the right
   convergence — but a deriva-ml consumer that already worked around
   the empty-string-as-None mismatch could see a behavior change.
   Likely fine; flag for the deriva-ml-side bump.

2. **`_coerce_pg_array` deletion.** Should the cleanup be in the same
   PR or a follow-up? Same PR is cleaner (no transitional state where
   two code paths handle the same case). Follow-up is safer
   (cleanup-revertible without losing the underlying fix). Bias
   toward same PR with the explicit "this becomes dead" rationale.

3. **`int4[]` / numeric arrays.** The parser
   (`[part.strip().strip('"') for part in inner.split(",")]`)
   produces `list[str]`. For an `int4[]` column the list is
   `["1","2","3"]`, not `[1,2,3]`. ERMrest's wire format accepts
   either (JSON typing is loose) but a strict consumer would care.
   Open: should the parser cast to the element type? The audit
   recommended not to over-engineer; passing through as strings
   matches `_coerce_pg_array`'s current behavior. Defer until a
   numeric-array consumer files a bug.

4. **`schema_io` round-trip.** Verify that with the fix in place, a
   bag write → bag read → bag write of a vocab table preserves the
   `text[]` declaration and the array values across two
   serialisation hops. Spot-check: `SQL_TO_ERMREST` at
   `schema_io.py:115` maps `ArrayAsJson → text[]`, so the type
   round-trips. Value round-trip should be Path A semantics on the
   second hop (the cell holds a JSON array, the read gives a list,
   the next write writes the list as a JSON array). Not reproduced
   live in this audit.

5. **Documentation pass.** The `deriva-bag` user-guide section in
   deriva-py docs and any deriva-ml-side denormalization.md page
   that mentions array columns should be re-read after the fix to
   confirm no stale claim survives. (Searches in this audit for
   "vocab", "Synonyms", "array", "text[]" in the relevant doc
   files turned up no problematic claims, but a docs pass with the
   fix in hand is worth one engineer-hour.)

---

## 11. Limitations of this audit

1. **In-memory reproduction only.** I exercised `ArrayAsJson`
   against a synthetic `:memory:` SQLite table; I did not run a
   full `BagBuilder → bag-on-disk → DataLoader → BagDatabase`
   pipeline end-to-end. The static frames (`csv.DictReader` yields
   `str`, `SqliteWriter` does no coercion, `JSON.bind_processor`
   runs `json.dumps`) are all reproduced from the actual code paths
   the live pipeline takes; the failure mode would be identical.
2. **Single SQLAlchemy version.** Verified against the version in
   `/Users/carl/GitHub/deriva-py/.venv/`
   (`sqlalchemy/sql/sqltypes.py` at
   line 2778). Different SQLAlchemy versions may inline a different
   bind processor; the `json.dumps` path is the standard and has
   been stable for a long time, but a major SQLAlchemy bump is the
   one risk vector for the docstring claim to spontaneously start
   working without code change.
3. **PG-literal-parser edge cases not stress-tested.** Empty arrays
   (`{}`), unquoted simple elements (`{a,b}`), and quoted elements
   (`{"a","b"}`) covered. NULL elements (`{a,NULL,b}`), embedded
   commas (`{"a,b",c}`), and escaped quotes (`{"a\\"b"}`) not
   exercised. `_coerce_pg_array`'s comment acknowledges the same
   gap — this audit doesn't widen it.
4. **Did not benchmark.** `process_bind_param` runs per row per
   array column. For a vocab with millions of rows the parsing
   overhead matters; in practice vocab tables are 10s to 100s of
   rows. Not worth a benchmark for the recommended fix.
5. **deriva-ml-side bump not exercised.** Once the fix lands in
   deriva-py, the lockfile bumps in deriva-ml and the model
   template need to follow. Not part of this audit's scope.
6. **Multi-process / concurrent writes not considered.** Single-
   writer assumption. Bag CSVs are static; no concurrency concern.
