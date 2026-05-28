# SQLite array-binding bug in denormalizer cache populate

**Investigator:** sqlite array-binding audit (research-only)
**Date:** 2026-05-28
**Scope:** deriva-ml @ HEAD on e2e-test/2026-05-28, deriva-py @ HEAD
(both checked into the e2e worktree). Reproduced live against
catalog 27 on dev-localhost.
**Mode:** No code changes. Every runtime claim is reproduced.

## 1. TL;DR

When the denormalizer's live-catalog cache populate path fetches rows
from any ERMrest **vocabulary table** (`Vocabulary.Synonyms` is
declared `text[]`), the `INSERT … VALUES (?, …, ?, ?)` step
crashes with `sqlite3.ProgrammingError: Error binding parameter 10:
type 'list' is not supported`.

Root cause is **upstream of deriva-ml in deriva-py**:
`deriva.bag._column_types.ERMREST_TO_SQL` has **no entry for any
array typename** (`text[]`, `int4[]`, etc.), so
`sql_type_for_ermrest()` silently falls back to `String` for every
array column when `SchemaBuilder` creates the SQLite mirror. The
SQLite column lands as `TEXT`; the row data the ERMrest server
returns for that column lands as a Python `list`; SQLAlchemy's
SQLite driver cannot bind a `list` to a `TEXT` parameter; the
INSERT raises.

Bind site: `deriva_ml/local_db/paged_fetcher.py:558` —
`conn.execute(stmt, projected)` inside `PagedFetcher._insert_rows`.

Scope on catalog 27: **all seven vocabulary tables** (`Image_Class`,
`Feature_Name`, `Asset_Type`, `Asset_Role`, `Execution_Status`,
`Workflow_Type`, `Dataset_Type`). The bug is generic to "any table
with an array-typed column" — `Image_Class.Synonyms` is what trips
on this catalog because it's the vocab the e2e fix-pass call shape
asks for, but every vocab table is equally broken.

Affected public methods: `Denormalizer.as_dataframe`,
`Denormalizer.as_dict`, and (transitively) every `Dataset` /
`DatasetBag` wrapper that calls them, **when** `include_tables`
contains a vocab table or the join-path traverses one. `columns()`
is safe (model-only, no fetch). `describe()` is safe (dry-run; planner
only).

**Recommended fix: option 1 (JSON-coerce list values, store as
TEXT in SQLite, decode on read).** This is the pattern deriva-py's
ERMrest `Type.sqlite3_ddl()` already specifies ("json" for
`ArrayType`); the bag's `_column_types.py` module just never
implemented it. Fix is small (one new `TypeDecorator` class plus
one dict entry in `ERMREST_TO_SQL`), one file
(`deriva/bag/_column_types.py`), with a single matching test fixture
addition in `deriva-py/tests/`.

**Pre-existing, not a regression.** `git log -S "text[]"` against
`deriva/bag/_column_types.py` returns no hits — the `ERMREST_TO_SQL`
table has never contained an array entry since the file was created.

## 2. Reproduction

### Minimal reproducer

```python
from deriva_ml import DerivaML
from deriva_ml.dataset import Dataset
from deriva_ml.local_db.denormalizer import Denormalizer

ml = DerivaML(hostname="localhost", catalog_id="27", use_minid=False)
ds = Dataset(ml, "VAP")                   # cifar10_small_labeled_split
d = Denormalizer(ds, version="0.1.0.post1.dev1")

# Fails with sqlite3.ProgrammingError on the Image_Class INSERT.
d.as_dataframe(include_tables=["Image", "Image_Class"], row_per="Image")
```

### Full stack trace

```
File ".../deriva_ml/local_db/denormalize.py", line 517, in _populate_from_catalog
    _populate_from_catalog_inner(...)
File ".../deriva_ml/local_db/denormalize.py", line 628, in _populate_from_catalog_inner
    fetcher.fetch_by_rids(
        table=qualified, rids=str_rids,
        target_table=target_orm.__table__, rid_column=fk_column_on_target,
    )
File ".../deriva_ml/local_db/paged_fetcher.py", line 316, in fetch_by_rids
    n += self._insert_rows(target_table, rows)
File ".../deriva_ml/local_db/paged_fetcher.py", line 558, in _insert_rows
    result = conn.execute(stmt, projected)
File ".../sqlalchemy/engine/default.py", line 949, in do_executemany
    cursor.executemany(statement, parameters)
sqlalchemy.exc.ProgrammingError: (sqlite3.ProgrammingError) Error binding
    parameter 10: type 'list' is not supported
[SQL: INSERT INTO "e2e-test-20260528"."Image_Class"
    ("RID", "RCT", "RMT", "RCB", "RMB", "ID", "URI",
     "Name", "Description", "Synonyms")
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT ("RID") DO NOTHING]
[parameters: [('3ST', '2026-05-28 16:51:46.281993', ..., 'airplane',
              'Fixed-wing aircraft', ['plane', 'aeroplane']),
             ('3SW', ..., 'automobile', 'Motor vehicle with four wheels',
              ['car', 'auto']),
             ('3SY', ..., 'bird', 'Feathered flying vertebrate', []),
             ...
             ('3TC', ..., 'truck', '...', ['lorry'])]]
```

The 10th positional parameter for every row is the `Synonyms` value —
a Python `list[str]` (the ERMrest server returns array columns as
JSON arrays, which `deriva-py`'s `_get` deserialises into Python
lists). Empty arrays come through as `[]`, populated ones as
`['syn1', 'syn2']`.

### Public methods affected (verified live)

| Method | Result | Why |
|---|---|---|
| `Denormalizer.columns(...)` | OK | Model-only; calls `_prepare_wide_table` planner without `_populate_from_catalog`. |
| `Denormalizer.describe(...)` | OK | Dry-run; planner-only path; no fetch. `warnings == []` — the dry-run invariant doesn't surface this. |
| `Denormalizer.as_dataframe(...)` | FAIL | Calls `_denormalize_impl` → `_populate_from_catalog` → `fetcher.fetch_by_rids` → `_insert_rows`. |
| `Denormalizer.as_dict(...)` | FAIL | Same path as `as_dataframe`. |
| `Dataset.denormalize_as_dataframe / as_dict` | FAIL | 4-line wrappers around `Denormalizer.as_*`. |
| `DatasetBag.denormalize_as_dataframe / as_dict` | Not reproduced live | Bag mode goes through `DataLoader` not `_populate_from_catalog`; CSV→SQLite path likely has its own behaviour (see §5). |

### What does NOT trip the bug

Verified live:

- `include_tables=["Image"]` alone — `df.shape == (434, 6)`. No vocab
  table is on the join path, so the populate walk never inserts into
  a table with an array column.
- Any `include_tables` that doesn't transitively walk through a vocab
  table.

The trigger is structural, not value-shaped: the bug fires the moment
`_populate_from_catalog_inner` is asked to load a table whose ERMrest
definition has at least one `text[]` / `*[]` column AND whose actual
catalog rows contain a non-NULL value for that column. (Catalog 27's
`Image_Class` ships with synonyms on 7 of 10 terms; the bug fires on
the very first INSERT batch.)

## 3. Root-cause analysis

### Where the SQLite column type is declared

`/Users/carl/GitHub/deriva-py/deriva/bag/schema.py:693`:

```python
database_column = SQLColumn(
    name=c.name,
    type_=sql_type_for_ermrest(c.type),   # <-- here
    comment=c.comment,
    default=c.default,
    primary_key=is_pk,
    nullable=c.nullok if is_pk else True,
)
```

`sql_type_for_ermrest()` lives in
`/Users/carl/GitHub/deriva-py/deriva/bag/_column_types.py:202` and is a
one-liner:

```python
def sql_type_for_ermrest(deriva_type):
    return ERMREST_TO_SQL.get(deriva_type.typename, String)
```

The lookup uses `deriva_type.typename`, which for an array column is
`"text[]"` / `"int4[]"` / etc. (see
`deriva/core/ermrest_model.py:3392-3402` for the `builtin_types`
construction). `ERMREST_TO_SQL` (lines 178-199 of `_column_types.py`)
has **19 entries** — `boolean`, `date`, scalar numerics, scalar
strings, `json`, `jsonb`, the `ermrest_*` system types. **Zero
entries for any `*[]` array typename, and no `is_array` branch.**
Result: every array column silently falls back to `String`.

The corresponding column on the SQLite side is therefore declared as
`TEXT`. SQLAlchemy's SQLite dialect can bind `str`, `int`, `float`,
`bytes`, `None`, and a handful of others to a `TEXT` parameter via
`sqlite3`'s built-in adapters — but **not `list`**. The bind step at
`paged_fetcher.py:558` raises:

```python
stmt = sqlite_insert(target_table).on_conflict_do_nothing(index_elements=["RID"])
with self._engine.begin() as conn:
    result = conn.execute(stmt, projected)   # <-- raises here
```

`projected` is a list of dicts, each dict the row dict the server
returned (filtered to just the columns the SQLite mirror declares).
The server's serialisation of `text[]` is a JSON array, which
`deriva.core` deserialises into a Python `list` — and that's what the
INSERT tries to bind.

### Why no coercion code fires

Three places turn ERMrest columns into SQLAlchemy columns:

1. **`SchemaBuilder._create_tables`** (the one this audit hit).
   Uses `sql_type_for_ermrest()`. No array branch.
2. **`BagDatabase` reflect-from-SQLite path**
   (`deriva/bag/database.py`). Uses the same
   `ERMREST_TO_SQL` table (since the dedup at
   `_column_types.py`). No array branch.
3. **`schema_io.ermrest_json_to_metadata`** (the bag-write
   lossless round-trip,
   `/Users/carl/GitHub/deriva-py/deriva/bag/schema_io.py`). Has its
   own type table (`SQL_TO_ERMREST`) and `JSON` does appear in it
   (line 113), but for **`jsonb`**, not for any `*[]` typename. No
   array branch either.

There is also `Type.sqlite3_ddl()` in
`/Users/carl/GitHub/deriva-py/deriva/core/ermrest_model.py:3373-3375`
which **correctly** returns `'json'` for `ArrayType`:

```python
class ArrayType(Type):
    def sqlite3_ddl(self) -> str:
        return 'json'
```

But this method is **only called from**:

```
$ grep -rn "sqlite3_ddl" /Users/carl/GitHub/deriva-py /Users/carl/GitHub/DerivaML/deriva-ml /Users/carl/GitHub/DerivaML/deriva-mcp-core
deriva/core/ermrest_model.py:3322:    def sqlite3_ddl(self) -> str:
deriva/core/ermrest_model.py:3354:    def sqlite3_ddl(self) -> str:
deriva/core/ermrest_model.py:3373:    def sqlite3_ddl(self) -> str:
```

— **the method is defined three times and called zero times.** The
ERMrest model knows that array columns should be SQLite `json`, but
no consumer reads that signal.

The `_column_types.py` module already imports SQLAlchemy `JSON`
(line 54) and uses it for the `json` / `jsonb` typenames (lines
186-187). SQLAlchemy's `JSON` type, when run against SQLite,
serialises Python `list` / `dict` values to JSON-encoded TEXT on
bind and deserialises them back on result-row read — which is
exactly the behaviour the array case needs. **The fix is to make
`ERMREST_TO_SQL` map every `*[]` typename (or all array types
generically) to `JSON`.**

### Concrete bind site quote

`paged_fetcher.py:520-564` — the bind call is at line 558,
inside `PagedFetcher._insert_rows`:

```python
def _insert_rows(self, target_table: Table, rows: list[dict[str, Any]]) -> int:
    ...
    for r in rows:
        if "RID" not in r:
            raise ValueError(...)
    cols = {c.name for c in target_table.columns}
    projected = [{k: v for k, v in r.items() if k in cols} for r in rows]
    if not projected:
        return 0
    stmt = sqlite_insert(target_table).on_conflict_do_nothing(index_elements=["RID"])
    with self._engine.begin() as conn:
        result = conn.execute(stmt, projected)   # <-- raise site
```

The raise site IS in `PagedFetcher._insert_rows`, not in
`_populate_from_catalog_inner` itself. The first agent's summary
("the bug is in `_populate_from_catalog_inner`") was correct about
the call chain but off by one frame.

## 4. Scope

### Tables on catalog 27 with array columns

All discovered by walking `catalog.getCatalogModel().schemas` and
filtering on `col.type.is_array`:

| Schema | Table | Column | Type |
|---|---|---|---|
| `deriva-ml` | `Feature_Name` | `Synonyms` | `text[]` |
| `deriva-ml` | `Asset_Type` | `Synonyms` | `text[]` |
| `deriva-ml` | `Asset_Role` | `Synonyms` | `text[]` |
| `deriva-ml` | `Execution_Status` | `Synonyms` | `text[]` |
| `deriva-ml` | `Workflow_Type` | `Synonyms` | `text[]` |
| `deriva-ml` | `Dataset_Type` | `Synonyms` | `text[]` |
| `e2e-test-20260528` | `Image_Class` | `Synonyms` | `text[]` |

**Seven total.** Every vocabulary table — the standard Deriva
vocabulary shape uses `Synonyms text[]` as part of the controlled-
term model. The user-defined `Image_Class` vocabulary inherits this.

No non-vocabulary array columns exist on this catalog. The model-
template schema has no `text[]` / `int[]` on `Image`, `Subject`,
features, asset tables, etc. That's coincidental — any future
caller could declare an array column on any table.

### deriva-ml functions that depend on the broken path

`_populate_from_catalog` is called from exactly one place:
`_denormalize_impl` at `denormalize.py:373`. Every public surface
that flows through `_denormalize_impl` with `source == "catalog"` is
affected:

- `Denormalizer.as_dataframe` (broken)
- `Denormalizer.as_dict` (broken)
- `Dataset.denormalize_as_dataframe` / `denormalize_as_dict` —
  4-line wrappers around the above (broken)
- `Denormalizer.describe` — calls `_denormalize_impl` only for
  anchor classification when "needed", and the dry-run invariant
  swallows exceptions into the `warnings` field. Verified live:
  `warnings == []` and `estimated_row_count.total = None` with a
  "anchor downstream" reason. So `describe` is **silently** less
  useful (it falls back to the planner-only path) but doesn't raise.
  This is the audit's most concerning blind spot: a user calling
  `describe` to vet a plan won't see that the materialise will
  fail.
- `Denormalizer.columns` — model-only path, no fetch, **safe**.

For `DatasetBag` (`source == "local"`), the bag is built by
`deriva.bag.builder.BagBuilder` → `DataLoader` → CSV pipeline. The
CSV→SQLite step has its own array handling story (see §5). I did
not reproduce the bag-mode case live; that's a Limitation below.

### What about feature tables, joined paths?

When the planner picks a join path that **traverses** a vocab table
even if the user didn't name it in `include_tables` (e.g.,
`include_tables=["Image", "Image_Classification"]` resolves to
`["Image", "Execution_Image_Image_Classification"]` and the join
walks through `Image_Class` as a transparent bridge), the populate
walk fetches **every table in the path**, including the transparent
ones. So the bug fires for any include set whose join path crosses
a vocab table — which on this catalog is most non-trivial shapes.
Verified: `include_tables=["Image"]` alone succeeds (no vocab on
path); `include_tables=["Image", "Image_Class"]` fails (vocab on
path).

## 5. Comparison with bag-mode (catalog_loader / DataLoader)

The bag-mode write path
(`/Users/carl/GitHub/deriva-py/deriva/bag/catalog_loader.py`) faces
the **inverse** problem: bag CSVs persist `text[]` columns as
PostgreSQL literal-array form (`{a,b}` or `{}`), and the loader has
to coerce those literals into JSON arrays for the ERMrest wire
format. The coercion lives in `_coerce_pg_array`
(lines 1207-1230 of `catalog_loader.py`):

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

And the caller at `catalog_loader.py:1334-1338`:

```python
# (text[] columns on the bag side arrive as PG literal-array
# (``{a,b}``) into real JSON arrays for the wire.
array_columns = [
    c.name
    for c in table.column_definitions
    if getattr(c.type, "is_array", False)
]
```

So in bag mode:

1. **Bag-write path**: ERMrest server → CSV file. ERMrest emits a
   JSON array; the CSV writer turns it into the PG literal `{a,b}`
   string for the file. (Need not be SQLAlchemy-mediated; the bag
   walker writes raw CSV.)
2. **Bag-read path** (CSV → SQLite via `DataLoader`): the SQLite
   column is declared as `String` (same bug-source as
   `SchemaBuilder`; same `_column_types.py` table), and the CSV row
   carries the PG literal as a string. So **the bag CSV→SQLite step
   loads the literal as a plain string** — no `list` value, no
   binding error, but the SQLite cell contains `"{a,b}"`, not
   `["a","b"]`. The bug is **latent**, hidden by the fact that CSVs
   never carry Python `list` values.
3. **Bag → catalog ingest**: `_coerce_pg_array` runs on the way
   out, turning `"{a,b}"` back into `["a","b"]` for the JSON ingest.

The denormalizer's **live catalog path** skips the CSV detour. The
ERMrest server returns JSON arrays directly to deriva-py's HTTP
client, which deserialises into Python `list` values, which then
flow straight into `_insert_rows` and hit the SQLite bind boundary
where `list` is not supported.

So bag mode doesn't "solve" this problem — it side-steps it by
storing array values as PostgreSQL literal strings in SQLite. That's
a workaround, not a solution: any consumer that reads the SQLite
mirror back and expects Python `list` values gets a PG literal
string instead, which is a different latent bug (silent type drift
on the read side).

**The right fix for both paths is to declare array columns as
`JSON` in the SQLite schema** — then SQLAlchemy `JSON` does the
right thing on both sides (Python `list` ↔ JSON-encoded TEXT) and
neither the live nor the bag path needs a string-detour workaround.
`catalog_loader._coerce_pg_array` becomes dead code on the way out
(values are already lists in the SQLite mirror) — that's a separate
cleanup, but the fix doesn't require it.

## 6. Recommended fix

**Option 1 (JSON-coerce at the type boundary).**

Implementation outline (single file, ~10 LoC):

In `/Users/carl/GitHub/deriva-py/deriva/bag/_column_types.py`, add:

```python
class ArrayAsJson(TypeDecorator):
    """Serialise Python list values as JSON-encoded TEXT for SQLite.

    ERMrest emits array columns (``text[]``, ``int4[]``, ...) as
    JSON arrays, which deriva-py deserialises into Python lists. SQLite
    has no native array type and SQLAlchemy's SQLite dialect cannot
    bind a list to a TEXT column. JSON-encode on write, decode on
    read, so callers see ``list`` end-to-end.
    """
    impl = JSON
    cache_ok = True
```

And the `sql_type_for_ermrest` helper grows an array branch:

```python
def sql_type_for_ermrest(deriva_type):
    if getattr(deriva_type, "is_array", False):
        return ArrayAsJson
    return ERMREST_TO_SQL.get(deriva_type.typename, String)
```

(Or, since `SQLAlchemy.JSON` already does list-round-tripping out of
the box on SQLite, the `ArrayAsJson` decorator is optional — just
returning `JSON` directly from the array branch may be sufficient.
A `TypeDecorator` wrapper is the safer shape because it gives a
named class for the bag's lossless round-trip
(`schema_io.SQL_TO_ERMREST`) to find on the inverse pass.)

### Why option 1 over options 2 / 3

**Option 2 (drop array columns from the local cache).** Surfaces
the problem to users: the denormalized wide table loses `Synonyms`
in the output. For a vocab-projecting denormalize this is
user-visible data loss. Rejected.

**Option 3 (delimiter-joined string).** Lossy on synonyms
containing the delimiter, asymmetric with the ERMrest wire format,
and breaks the catalog_loader's `_coerce_pg_array` round-trip.
Rejected.

**Option 1 (JSON via `SQLAlchemy.JSON`).** The serialisation is
already in SQLAlchemy. `deriva.core.Type.sqlite3_ddl()` already
*specifies* this is the right SQLite type for arrays. The fix is
adding what was always intended. No data loss; symmetric on
read/write; integrates with the existing `JSON` entries for `json`
and `jsonb`.

### Fix is small (single function in single file)

- One edit: `deriva/bag/_column_types.py:202` —
  `sql_type_for_ermrest` grows an `is_array` branch (or
  `ERMREST_TO_SQL` grows an `ARRAY_TYPENAME_PATTERN` lookup; either
  approach works).
- One companion: a unit test in deriva-py exercising
  `SchemaBuilder` against a `text[]` column with non-empty values.
- No deriva-ml changes required. The bag pipeline picks up the fix
  automatically (same module).
- One companion cleanup later: `_coerce_pg_array` in
  `catalog_loader.py` becomes dead code if the bag CSV writer
  starts producing JSON arrays instead of PG literals — but
  that's a follow-up, not part of this fix. The fix above doesn't
  break either path (in fact it makes bag-load consistent with the
  live path).

The lock file in `deriva-ml/uv.lock` would need to be bumped to a
new deriva-py version once the fix lands. Mechanical.

### Out-of-scope adjacencies (do NOT bundle)

- **`Type.sqlite3_ddl()` triple-defined, zero-callers.** Worth a
  separate cleanup PR in deriva-py: either delete the methods (the
  fix above doesn't use them), or wire them up as the canonical
  type-translation source and retire `ERMREST_TO_SQL`. Not in
  scope for the fix.
- **`_coerce_pg_array` → dead code.** Same situation as above.
- **`schema_io.SQL_TO_ERMREST` mapping for ArrayAsJson → text[].**
  Needs an inverse entry so the lossless schema round-trip works on
  array columns. Should be in the same fix PR (one-line
  addition).

## 7. Test gap analysis

### Existing tests that touch array columns

```
$ grep -l "text\[\]\|Synonyms\|is_array\|ArrayType" \
    /Users/carl/GitHub/DerivaML/deriva-ml/tests/dataset/*.py \
    /Users/carl/GitHub/DerivaML/deriva-ml/tests/local_db/*.py
# (no results)
```

**Zero existing tests in deriva-ml exercise vocab-table
denormalization through `as_dataframe` or `as_dict`.** The
`tests/dataset/test_denormalize.py` tests only call
`get_denormalized_as_dataframe(include_tables=["Subject", "Image",
"Observation"])` — none of those tables have array columns.

The reference fixtures themselves DO contain array columns
(`tests/dataset/demo-catalog-schema.json` has 1 occurrence of
`text[]`, `deriva-ml-reference.json` has 6) — so the schema-side
tests, if they exercised SchemaBuilder against vocab tables with
non-empty rows, would catch this. They don't.

### What would catch this bug

A test that constructs a `Denormalizer` from a bag (or live)
fixture that includes a populated vocab table — `Asset_Type` is in
the demo schema and has rows in `tests/dataset/demo-catalog/data/
deriva-ml/Asset_Type.csv` (the standard `image_url`, `image_file`,
etc. taxonomy). A single test like:

```python
def test_denormalize_with_vocab_table(dataset_test, tmp_path):
    bag = build_bag_from(dataset_test, tmp_path)
    df = bag.get_denormalized_as_dataframe(
        include_tables=["Subject", "Asset_Type"]
    )
    # Verify Synonyms column round-trips as list, not as str / dropped.
    assert "Asset_Type.Synonyms" in df.columns
    assert all(isinstance(v, (list, type(None))) for v in df["Asset_Type.Synonyms"])
```

would have caught the live-catalog bug **and** the bag-CSV string-
detour (the bag path's Synonyms cells would come back as PG literal
strings, failing the `isinstance(v, list)` check).

A deriva-py-side analogue at the `_column_types.py` layer would be
even more targeted:

```python
def test_array_column_round_trip(model_with_text_array):
    builder = SchemaBuilder(model=model_with_text_array, ...)
    orm = builder.build()
    table = orm.find_table("Vocab")
    with orm.engine.begin() as conn:
        conn.execute(insert(table), [{"RID": "X", "Synonyms": ["a", "b"]}])
        out = list(conn.execute(select(table)))
    assert out[0].Synonyms == ["a", "b"]
```

### Coverage gap, not test-design gap

The denormalizer test suite uses a focused fixture (Subject/Image/
Observation) that omits the structural shape (vocab tables with
populated arrays) where the bug lives. It's a **fixture gap**, not
an architectural blind spot — the test framework itself supports
adding a vocab-inclusive case in a few lines.

## 8. Open questions

1. **Does bag-mode `DatasetBag.denormalize_as_dataframe` raise the
   same error on read?** Not reproduced live. Likely it returns
   strings (`"{plane,aeroplane}"`) rather than lists for vocab rows
   in the result DataFrame — different bug, same root cause. Worth
   a 5-minute follow-up to confirm.
2. **Are there any catalogs in the wild with non-vocab array
   columns?** I checked catalog 27. The deriva-ml stock vocab
   tables (`Feature_Name` etc.) all carry `Synonyms text[]` by
   construction, so any catalog using deriva-ml has the
   vulnerability. User-defined array columns on non-vocab tables
   would be additional triggers.
3. **Why does `describe()` not surface the future failure?**
   `describe()` runs through `_prepare_wide_table` (planner only)
   without invoking `_populate_from_catalog`. The describe dry-run
   honestly reports what the planner can see, but it has no way to
   know that materialisation will hit a downstream type-binding
   error. Adding a "would `as_dataframe` raise?" predictive check
   to `describe` is out of scope for this fix but worth noting as
   a UX gap — the dry-run invariant doesn't extend to
   materialisation-layer errors.
4. **Does the `paged_fetcher_ermrest` source layer do anything
   different from the JSON wire format?** I read
   `paged_fetcher.py` (the abstract base) but not
   `paged_fetcher_ermrest.py` (the concrete impl). The bind site
   raise IS at the abstract layer, so the answer doesn't change
   the diagnosis, but a follow-up could check whether the concrete
   client does any pre-coercion the abstract bind would benefit
   from.

## 9. Limitations of this audit

1. **Bag-mode denormalize not reproduced live.** I read enough
   `catalog_loader.py` to characterise the static type-translation
   pipeline (it shares `_column_types.py`, so it shares the bug
   structurally) but did not run a `DatasetBag` against a
   vocab-table denormalize. The bag-mode comparison in §5 is
   inference from code reading, not live observation.
2. **No deep dive into `schema_io.SQL_TO_ERMREST`.** I confirmed it
   doesn't have an array entry but didn't trace whether the
   lossless schema round-trip in builder mode silently drops array
   columns or fails. That's relevant for the cleanup follow-up,
   not the fix.
3. **No live verification of the proposed fix.** I described what
   the fix should look like but did not apply it locally and re-run
   the reproducer. The audit is research-only by the user's
   directive.
4. **One catalog only.** Reproduced against catalog 27
   (CIFAR-10 schema). Did not exercise CSA / CFDE / GPCR catalogs
   in production. The vocab-table case generalises trivially
   (every deriva-ml-loaded catalog has the same six core vocab
   tables); other shapes (non-vocab arrays, `int[]`, multi-dim
   arrays) were not exercised.
5. **No test of `Denormalizer.as_dict` row-by-row consumption
   pattern.** Verified it raises during populate, before any rows
   are yielded. A user who calls `as_dict()` and iterates may get
   the exception during iteration (depends on the executor's
   lazy-vs-eager semantics in `_denormalize_impl`); I did not
   probe the precise yield point.
6. **`PR #254 docstring fix` agent's note and `PR #55 model-
   template revision` agent's reproduction were noted but not
   independently reproduced** — I built my own minimal reproducer
   from the schema rather than copy theirs. Both prior agents'
   characterisations match what I see live.
