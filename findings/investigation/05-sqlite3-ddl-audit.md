# Investigation 05 — `sqlite3_ddl` audit in deriva-py

**Status:** research-only. No code changed. No PRs opened.
**Scope:** the chain of `sqlite3_ddl()` methods on `Table`, `Column`,
`Key`, `Type`, `DomainType`, `ArrayType` in
`deriva/core/ermrest_model.py`.
**Trigger:** finding 04 (sqlite array-binding bug) noted in passing
that `Type.sqlite3_ddl` is "triple-defined" with "zero callers" and
suggested either deleting it or wiring it up to replace
`ERMREST_TO_SQL`. This audit verifies both claims and evaluates the
two paths.

---

## 1. TL;DR

The prior audit was **wrong on both counts**:

- Not 3 definitions — **6**: `Table`, `Column`, `Key`, `Type`,
  `DomainType`, `ArrayType` each own a `sqlite3_ddl` method.
- Not "0 callers" — `Table.sqlite3_ddl` calls `Column.sqlite3_ddl`
  and `Key.sqlite3_ddl`; `Column.sqlite3_ddl` calls
  `Type.sqlite3_ddl`; `DomainType.sqlite3_ddl` calls
  `self.base_type.sqlite3_ddl`. There is one *external* entry point
  (`Table.sqlite3_ddl`) and the rest of the chain is internally
  reachable from it.

But the prior audit was **right on the bottom line**: across the
entire deriva-py codebase (`deriva/`, `tests/`, `docs/`) and the
seven DerivaML-org repos on this machine, **no caller of
`Table.sqlite3_ddl` exists**. The chain is reachable in principle but
nothing reaches it. All six methods were introduced in a single
commit (`ef3c6809`, 2024-10-25, PR #187) as "suitable for capturing
table dumps from ERMrest" — speculative infrastructure that was
never wired up. The follow-up SQLite mirror work (`BagDatabase`,
`_column_types.ERMREST_TO_SQL`, Jan 2026) went a different route:
SQLAlchemy types built from `ERMrest` column descriptions, not DDL
strings.

**Recommendation: delete all 6 methods + `Table.sqlite3_table_name`
(the sibling helper added in the same commit, also unused) in a
small follow-up PR against deriva-py.** Risk is low but not zero —
the methods *are* implicitly exposed via Sphinx `automodule
:members:` in `docs/api/deriva.core.rst`, so a third-party consumer
outside the DerivaML org could in principle have grown a
dependency on them in the 19 months since merge. The fix is small
and easy to revert, and the chain has zero internal value (no test
covers it; nothing in the bag SQLite-mirror path uses it).

**Do not pursue the "wire up as canonical" path.** `sqlite3_ddl`
returns DDL strings; `ERMREST_TO_SQL` returns SQLAlchemy
`TypeEngine` classes that decorate the round-trip CSV-to-Python
conversion. They encode different information in incompatible
forms; replacing `ERMREST_TO_SQL` with `sqlite3_ddl` would lose the
`TypeDecorator` machinery the bag mirror depends on for CSV
coercion.

---

## 2. Complete inventory

All in `/Users/carl/GitHub/deriva-py/deriva/core/ermrest_model.py`.
All six methods introduced in commit `ef3c6809` (Karl Czajkowski,
2024-10-25, "add Table.sqlite3_ddl() method that returns a SQL DDL
statement", PR #187).

| Class | Line | Signature | Returns | Public/Private | Most recent activity |
|---|---|---|---|---|---|
| `Table` | 2356 | `sqlite3_ddl(self, keys: bool=True) -> str` | full `CREATE TABLE IF NOT EXISTS "schema:name" (...)` DDL | public (no underscore) | `f0166a74` Carl Kesselman 2024-11-12 — fix missing open paren in CREATE template |
| `Column` | 2685 | `sqlite3_ddl(self) -> str` | column DDL fragment: `"name" type [NOT NULL]` | public | `ef3c6809` (introduction only) |
| `Key` | 2953 | `sqlite3_ddl(self) -> str` | `UNIQUE (col1, col2, ...)` constraint fragment | public | `ef3c6809` (introduction only) |
| `Type` (`RegularType`) | 3322 | `sqlite3_ddl(self) -> str` | DDL type string from a 12-entry typename→string map; falls back to `'text'` | public | `ef3c6809` (introduction only) |
| `DomainType` | 3354 | `sqlite3_ddl(self) -> str` | delegates to `self.base_type.sqlite3_ddl()` | public | `ef3c6809` (introduction only) |
| `ArrayType` | 3373 | `sqlite3_ddl(self) -> str` | constant `'json'` | public | `ef3c6809` (introduction only) |

Sibling helper added in the same commit, also unused:

| Class | Line | Signature | Returns | Public/Private |
|---|---|---|---|---|
| `Table` | 2349 | `sqlite3_table_name(self) -> str` | `"schema:name"` | public — called only by `Table.sqlite3_ddl` |

Docstring claims (verbatim):

- `Table.sqlite3_ddl`: "Return SQLite3 table definition DDL
  statement for this table. ... Caveat: this utility does not
  produce: column default expressions, foreign key constraint DDL.
  Both of these features are fragile in data export scenarios where
  we want to represent arbitrary ERMrest catalog dumps."
- `Column.sqlite3_ddl`: "Return SQLite3 column definition DDL
  fragment for this column."
- `Key.sqlite3_ddl`: "Return SQLite3 unique constraint DDL fragment
  for this key."
- `Type.sqlite3_ddl` / `DomainType.sqlite3_ddl` /
  `ArrayType.sqlite3_ddl`: "Return a SQLite3 column type DDL
  fragment for this type."

All methods follow Google-ish docstring style but none have
`Example:` blocks. None are documented in user-facing docs.

---

## 3. Internal call graph

```
Table.sqlite3_ddl(keys=True)            <-- only external-facing entry point
  for col in self.columns:
    -> Column.sqlite3_ddl()
         -> self.type.sqlite3_ddl()       (Type | DomainType | ArrayType)
              DomainType -> self.base_type.sqlite3_ddl()
                              (recursively to Type or another DomainType)
  if keys:
    for key in self.keys:
      -> Key.sqlite3_ddl()

Table.sqlite3_ddl also calls Table.sqlite3_table_name() (sibling helper).
```

`Column.sqlite3_ddl`, `Key.sqlite3_ddl`, and all three
`Type.sqlite3_ddl` overrides are reachable only via
`Table.sqlite3_ddl`. There is no other in-repo caller for any of
them.

Verification: `grep -rn "\.sqlite3_ddl\b"
/Users/carl/GitHub/deriva-py` returns exactly the 10 lines reported
in section 4 below — all definitions or internal calls within
`ermrest_model.py` itself.

---

## 4. External caller search

### 4.1 deriva-py (the repo that owns the methods)

```
$ grep -rn "sqlite3_ddl" /Users/carl/GitHub/deriva-py/
deriva/core/ermrest_model.py:2356:    def sqlite3_ddl(self, keys: bool=True) -> str:
deriva/core/ermrest_model.py:2369:        parts = [ col.sqlite3_ddl() for col in self.columns ]
deriva/core/ermrest_model.py:2371:            parts.extend([ key.sqlite3_ddl() for key in self.keys ])
deriva/core/ermrest_model.py:2685:    def sqlite3_ddl(self) -> str:
deriva/core/ermrest_model.py:2689:            self.type.sqlite3_ddl(),
deriva/core/ermrest_model.py:2953:    def sqlite3_ddl(self) -> str:
deriva/core/ermrest_model.py:3322:    def sqlite3_ddl(self) -> str:
deriva/core/ermrest_model.py:3354:    def sqlite3_ddl(self) -> str:
deriva/core/ermrest_model.py:3356:        return self.base_type.sqlite3_ddl()
deriva/core/ermrest_model.py:3373:    def sqlite3_ddl(self) -> str:
```

All 10 lines are inside the single file. **No tests** mention
`sqlite3_ddl` (`grep -rn sqlite3_ddl tests/` returns empty). **No
docs** mention it (`grep -rn sqlite3_ddl docs/` returns empty —
though the Sphinx `automodule` directive auto-exposes it, see §5).

### 4.2 Across DerivaML-org repos on this machine

```
$ for repo in deriva-mcp deriva-mcp-core deriva-ml deriva-ml-mcp \
              deriva-ml-model-template deriva-ml-model-template-e2e \
              deriva-ml-skills deriva-plugins deriva-skills; do
    echo "=== $repo ==="
    grep -rn "sqlite3_ddl" "/Users/carl/GitHub/DerivaML/$repo" 2>/dev/null \
      || echo "(no matches)"
  done

=== deriva-mcp ===              (no matches)
=== deriva-mcp-core ===         (no matches)
=== deriva-ml ===               (no matches)
=== deriva-ml-mcp ===           (no matches)
=== deriva-ml-model-template === (no matches)
=== deriva-ml-model-template-e2e ===
  findings/investigation/04-sqlite-array-binding-bug.md (the prior audit; not code)
=== deriva-ml-skills ===        (no matches)
=== deriva-plugins ===          (no matches)
=== deriva-skills ===           (no matches)
```

The only hits outside deriva-py are inside the **prior audit
document itself** (finding 04). No code in any DerivaML-org repo
imports or calls `sqlite3_ddl`.

---

## 5. Public API status

### 5.1 Explicit exports

- `deriva/__init__.py` is a single-line namespace-package shim:
  `__path__ = __import__('pkgutil').extend_path(__path__, __name__)`.
  Nothing exported at the top level.
- `deriva/core/__init__.py`: searched for `__all__` and `sqlite3_ddl`
  — no explicit mention. Names are reachable via attribute access
  on imported classes (`Table`, `Column`, `Type`, etc.) but not in
  any `__all__`.
- `ermrest_model.py` has no `__all__`. Every class and method that
  doesn't start with `_` is implicitly part of the public API
  surface.

### 5.2 Implicit Sphinx documentation

In `/Users/carl/GitHub/deriva-py/docs/api/deriva.core.rst:74`:

```
.. automodule:: deriva.core.ermrest_model
    :members:
    :undoc-members:
    :show-inheritance:
```

`:members:` with no allowlist means **every public method on every
class is auto-rendered into the published API docs.** All six
`sqlite3_ddl` methods and `sqlite3_table_name` are therefore
implicitly documented as part of the public deriva-py API. This is
the strongest evidence that "delete" is a (small) breaking-change
risk for unknown downstream consumers, since the published Sphinx
docs at the time of writing list these methods as available.

### 5.3 Other surfaces

- No type stubs (`*.pyi`) ship in deriva-py.
- `CHANGELOG.md` was empty when searched for "sqlite" (file does
  not exist at the path I checked — verified with `cd
  /Users/carl/GitHub/deriva-py && grep -i sqlite README.md
  CHANGELOG.md` returning no output. README has no SQLite mention.)
- No ADR or design doc references `sqlite3_ddl`. The only ADR
  present (`docs/adr/0001-bag-catalog-loader-conflict-and-system-content.md`)
  is about a different topic.
- `docs/design/` covers bag-related design but doesn't reference
  `sqlite3_ddl` directly.

**Net:** the methods are implicitly part of the public API surface
via Sphinx but have never been called out as a feature in any
release note, README, ADR, or design doc.

---

## 6. Historical purpose

### 6.1 Commit message (PR #187, `ef3c6809`)

> add Table.sqlite3_ddl() method that returns a SQL DDL statement
>
> This is a basic table definition with no foreign keys and no
> default expressions, suitable for capturing table dumps from
> ERMrest.
>
> Table names are generated as "schema:table" to handle the fact
> that ERMrest has named schemas but SQLite3 is mostly a single
> schema namespace per database file.

### 6.2 PR #187 thread

The only review comments on PR #187 (Karl Czajkowski → Bob
Schuler):

- robes: `columnd - is that a typo in comment?`
- karlcz: `yup`
- robes: `Looks good to me. I ran tests too.`

No discussion of intended call site. The "I ran tests too" refers
to the project's general test suite, not to a `sqlite3_ddl`-
specific test (none exists).

### 6.3 What replaced (would have replaced) it

15 months later (Jan 2026), Carl Kesselman added `BagDatabase` in
commit `0a0f88d`: "Add BagDatabase for schema-independent BDBag to
SQLite conversion." That commit introduced the **actual** SQLite-
mirror code path the platform uses today. It went via SQLAlchemy
`MetaData` + reflected `Table`/`Column` objects, **not** via
`Table.sqlite3_ddl` DDL strings.

A few months further on, the bag-audit-cleanup work (Spring 2026)
extracted the shared column-construction logic into
`deriva/bag/_column_types.py`, codifying `ERMREST_TO_SQL` and
`sql_type_for_ermrest()` as the canonical ERMrest-to-SQLAlchemy type
map. By then `sqlite3_ddl` had been in the codebase for 18 months
without ever being used.

The most plausible reading: `sqlite3_ddl` was speculative
infrastructure for a "dump ERMrest catalog to SQLite" feature that
was later built on a different foundation (SQLAlchemy reflection)
which never circled back to consume the DDL chain.

---

## 7. `ERMREST_TO_SQL` comparison

The prior audit's §6.2 suggested "wire up `Type.sqlite3_ddl` as the
canonical type-translation source and retire `ERMREST_TO_SQL`." This
section evaluates whether that swap is even possible.

### 7.1 What each one returns

`Type.sqlite3_ddl()` (regular type) — returns a **DDL string**:

```python
{
    'boolean': 'boolean',
    'date': 'date',
    'float4': 'real',
    'float8': 'real',
    'int2': 'integer',
    'int4': 'integer',
    'int8': 'integer',
    'json': 'json',
    'jsonb': 'json',
    'timestamptz': 'datetime',
    'timestamp': 'datetime',
}.get(self.typename, 'text')
```

12 typenames; unknown falls back to `'text'`.
`ArrayType.sqlite3_ddl` always returns `'json'`.
`DomainType.sqlite3_ddl` delegates to base type.

`ERMREST_TO_SQL` (`deriva/bag/_column_types.py:199`) — returns
**SQLAlchemy `TypeEngine` classes** (most of which are
`TypeDecorator` subclasses that *also* handle CSV string coercion):

```python
ERMREST_TO_SQL: dict[str, type[TypeEngine]] = {
    "boolean":      ERMRestBoolean,      # TypeDecorator over Boolean
    "date":         StringToDate,        # TypeDecorator over Date
    "float4":       StringToFloat,       # TypeDecorator over Float
    "float8":       StringToFloat,
    "int2":         StringToInteger,     # TypeDecorator over Integer
    "int4":         StringToInteger,
    "int8":         StringToInteger,
    "json":         JSON,
    "jsonb":        JSON,
    "timestamptz":  StringToDateTime,    # TypeDecorator over DateTime
    "timestamp":    StringToDateTime,
    "text":         String,
    "longtext":     String,
    "markdown":     String,
    "ermrest_rid":  String,
    "ermrest_rct":  StringToDateTime,
    "ermrest_rmt":  StringToDateTime,
    "ermrest_rcb":  String,
    "ermrest_rmb":  String,
}
```

19 typenames; unknown falls back to `String` via
`sql_type_for_ermrest()`. Array types route to `ArrayAsJson` (a
`TypeDecorator` over JSON) regardless of element type.

### 7.2 Are they equivalent?

**No.** They encode different information:

| Property | `Type.sqlite3_ddl` | `ERMREST_TO_SQL` |
|---|---|---|
| Output form | DDL string (`'integer'`) | Python class (`StringToInteger`) |
| Typenames covered | 12 | 19 (covers `text`, `longtext`, `markdown`, `ermrest_*`) |
| Fallback for unknown | `'text'` string | `String` class |
| Array handling | `'json'` string | `ArrayAsJson` class with CSV decoder |
| CSV-to-Python coercion | none (string is just DDL) | yes — every `TypeDecorator` runs `process_bind_param` to coerce CSV `str` → typed Python value |
| Round-trip metadata | none | `col.info` stash for lossless schema_io round-trip (see comment block in `_column_types.py:30-39`) |

The DDL string `'integer'` told SQLite "this column holds integers."
The `StringToInteger` class tells SQLAlchemy "when binding a value
to this column, if it's a `str`, parse it as an int; if it's empty,
treat as NULL." Without the second behavior, every CSV row loaded
into the bag SQLite mirror would error or store strings in
columns declared as `integer`.

### 7.3 Could one drive the other?

In theory, SQLAlchemy can parse DDL strings (`text("integer")`) into
column types, but that buys you only the SQLite affinity hint — not
the `TypeDecorator` machinery. So:

- **Replace `ERMREST_TO_SQL` with `sqlite3_ddl`?** No. You'd lose
  CSV coercion (`process_bind_param`), drop 7 of the 19 typenames
  the bag pipeline supports (`text`, `longtext`, `markdown`,
  `ermrest_rid`, `ermrest_rct`, `ermrest_rmt`, `ermrest_rcb`,
  `ermrest_rmb`), and the fallback to "text" instead of `String`
  would mean unknown columns lose their type-decoration too.
- **Use `sqlite3_ddl` *alongside* `ERMREST_TO_SQL` for some
  other purpose?** No documented purpose exists. The "dump
  arbitrary ERMrest catalog as DDL text" feature implied by the
  PR #187 description was never built — `BagDatabase` does the
  job through SQLAlchemy reflection, never emitting DDL strings.

**Conclusion: the two are not equivalent and not interchangeable.
The "replace" path in the prior audit was an overstatement.**

---

## 8. Recommended action

**Recommendation: full deletion.**

Delete the following from
`/Users/carl/GitHub/deriva-py/deriva/core/ermrest_model.py`:

| Lines (approx) | What | Why safe to delete |
|---|---|---|
| 2349–2354 | `Table.sqlite3_table_name` | Called only by `Table.sqlite3_ddl`. |
| 2356–2379 | `Table.sqlite3_ddl` | No caller anywhere. |
| 2685–2693 | `Column.sqlite3_ddl` | Caller is `Table.sqlite3_ddl` (also deleted). |
| 2953–2956 | `Key.sqlite3_ddl` | Caller is `Table.sqlite3_ddl` (also deleted). |
| 3322–3336 | `Type.sqlite3_ddl` | Caller is `Column.sqlite3_ddl` (also deleted). |
| 3354–3356 | `DomainType.sqlite3_ddl` | Caller is `Column.sqlite3_ddl` (also deleted). |
| 3373–3375 | `ArrayType.sqlite3_ddl` | Caller is `Column.sqlite3_ddl` (also deleted). |

Total: ~7 contiguous-ish hunks, ~70 lines of code in one file. No
tests to update (none exist). No docs to update (Sphinx auto-doc
will silently drop the methods).

### Rationale

1. **No callers, not even speculatively.** 19 months in HEAD, zero
   in-repo callers (other than the chain calling itself), zero
   external callers in any DerivaML-org repo on this machine, no
   tests, no documentation example. The methods were added to PR
   #187 with a "suitable for capturing table dumps from ERMrest"
   description that never materialised into a feature.
2. **The replacement path doesn't exist.** §7 above shows `sqlite3_ddl`
   and `ERMREST_TO_SQL` encode incompatible information; the audit-04
   "wire it up as canonical source" suggestion is not technically viable.
3. **Maintenance cost is real.** The methods carry a 12-entry typename
   map that's a strict subset of `ERMREST_TO_SQL`'s 19. Any new
   ERMrest typename added to the catalog ecosystem (it has happened
   — `markdown`, `longtext`, `ermrest_*` since the original 12-entry
   map was written) would either need to be added in two places or
   silently fall back to `'text'` in `sqlite3_ddl` while doing the
   right thing in `ERMREST_TO_SQL`. Deletion eliminates the
   divergence risk.
4. **Cheap to revert.** All six methods came from a single 85-line
   commit. If a downstream user surfaces a need, restoring the
   commit is trivial. The "we might want this someday" cost is at
   most one commit revert.

### Not recommended

- **Wire `Type.sqlite3_ddl` as canonical to replace
  `ERMREST_TO_SQL`.** Loses CSV coercion (§7.3). Don't do this.
- **Leave alone.** The maintenance cost (silently growing
  divergence with `ERMREST_TO_SQL`) outweighs the speculative
  preservation value.
- **Partial deletion (keep some, drop others).** The chain is
  internally entangled — deleting the entry point makes everything
  downstream unreachable, and deleting downstream pieces breaks the
  entry point. It's an all-or-nothing chain.

---

## 9. Risk assessment for deletion

**Risk level: low, but not zero.**

### 9.1 Worst-case impact

A third-party deriva-py consumer outside the DerivaML org has
imported `Table.sqlite3_ddl` (or any other method in the chain) into
their own code. On upgrade, that import succeeds (the attribute is
still on the class until deletion) but the next call site raises
`AttributeError`. Their bug, but a deriva-py upgrade triggered it.

### 9.2 How likely is that?

- **No callers in any DerivaML-org repo** — established in §4.
- **No public release note** ever mentioned this method
  (CHANGELOG/README/ADR/design doc: empty for `sqlite3_ddl`).
- The methods *are* implicitly exposed via Sphinx `automodule
  :members:` (§5.2), so a third party browsing the API docs would
  see them.
- 19 months between merge (Oct 2024) and now (May 2026) — long
  enough for someone to have spotted it. Short enough that the
  population of third-party consumers using a niche method named
  `sqlite3_ddl` is unlikely to be large.
- The method does a narrow, easily-replaced job (generate SQLite
  DDL from an ERMrest schema). A consumer hitting `AttributeError`
  can rewrite a 30-line replacement in an afternoon — this is not
  a load-bearing API.

### 9.3 How would we find out we broke someone?

We wouldn't, until they filed an issue. Mitigation:

- Mention removal in the deriva-py changelog/release notes for the
  release that drops it.
- Tag the deletion PR with a clear `BREAKING:` prefix.
- Optional: deprecate first (raise `DeprecationWarning` with a
  pointer to `deriva.bag._column_types`) for one minor release, then
  delete. This adds ~3 LOC per method and one release cycle of
  patience. Given the zero-known-callers picture, this is probably
  over-engineering, but it is the conservative path.

### 9.4 Overall

Risk is low enough that I'd recommend just deleting in a single PR
with a `BREAKING:` flag in the title and a clear changelog entry. If
the project prefers extra caution, deprecate-first is available.

---

## 10. Open questions

- **Was there a private branch or downstream fork that planned to
  consume `sqlite3_ddl`?** I checked `git branch -a` on deriva-py
  (90+ branches) and saw nothing whose name suggests "use the
  sqlite ddl chain"; the original `sqlite_ddl` branch was the one
  that *added* the methods, not a consumer. But I can't rule out
  unmerged private forks outside this machine.
- **PyPI download stats for deriva-py?** Couldn't measure unique
  third-party callers from this audit; would require querying
  PyPI's download metrics + GitHub code search across the public
  ecosystem. Worth a one-off check before merging the deletion PR.
- **Are there any deriva-py consumers in the `informatics-isi-edu`
  GitHub org beyond what's checked out on this machine?** Not
  audited. A `gh search code "sqlite3_ddl" --owner informatics-isi-edu`
  would close this gap; I did not run it as part of the 30-minute
  budget.

---

## 11. Limitations of this audit

- **Filesystem scope only.** External callers were searched across
  the seven DerivaML-org repos plus the e2e worktree on this
  machine. Public deriva-py consumers (PyPI, GitHub) were *not*
  searched. The "zero callers" claim is therefore strict for the
  local DerivaML ecosystem and provisional for the wider Python
  ecosystem.
- **No PyPI download/dependency analysis.** Would tighten the risk
  estimate in §9.
- **No runtime test.** I did not invoke `Table.sqlite3_ddl()`
  against a live ermrest catalog to verify it currently produces
  syntactically-valid DDL. The November 2024 missing-paren bug fix
  (`f0166a74`) is evidence that at least one bug was caught in
  hand-testing once. I did not check whether other bugs lurk in the
  current implementation (e.g., whether `key.sqlite3_ddl` produces
  the right `UNIQUE` constraint syntax for multi-column keys with
  reserved-word column names). For a "delete" recommendation this
  doesn't matter — bugs in dead code are not worth fixing. For a
  hypothetical "keep and start using" recommendation, the methods
  would need testing first.
- **Git-blame range only.** I did not exhaustively walk every
  commit between `ef3c6809` (Oct 2024) and HEAD looking for
  references that may have been added and removed in the same
  window. The two `git log -S "sqlite3_ddl"` searches (across all
  refs) returned only the introduction commit and the one comment-
  typo / paren-fix follow-ups, so this seems thorough enough.

---

*Audit author: Claude (Opus 4.7), 2026-05-28, run-mode = research-only.*
*Inputs: deriva-py at HEAD (current `2.0-dev`), seven DerivaML-org
sibling repos at HEAD, prior audit document
`04-sqlite-array-binding-bug.md`.*
