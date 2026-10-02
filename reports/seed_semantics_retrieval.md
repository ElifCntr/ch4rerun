# Seed semantics retrieval, two code states

Retrieval performed 24 September 2026; written to file 2 October 2026.
Read-only, from the git objects at two commits. Quotes and locations only. No
interpretation is offered and no verdict is drawn.

| label | commit | what executed it |
| --- | --- | --- |
| **A** | `a8c243c` | the 21 standing runs |
| **B** | `fdaa841` | the three post-flip-fix scratch r3d runs |

`a8c243c` is "Training harness: dataset, region-level AP, streaming
evaluation, training loop", 21 August 2026. `fdaa841` is "Add scratch mode for
the from-scratch appendix baseline", 27 August 2026.

`src/seeding.py` and `tools/materialise_tubelets.py` are byte-identical
between the two states; `git diff a8c243c fdaa841` over each returns nothing.
All differences are confined to `src/tubelets.py` and `tools/train_arm.py`.
`src/arms.py` does not exist at A.

Every quotation below was checked against the git object at the commit and
line given, including the absence assertions in sections 2 and 3.

## 1. Every call that seeds anything, and what it receives

### Entry point

Both states call one seeding function, receiving the run seed itself,
underived and unmodified.

- A `tools/train_arm.py:115`, B `tools/train_arm.py:117`

      seed_everything(args.seed, True, False)

- A and B `tools/train_arm.py:52`

      from src.seeding import seed_everything  # noqa: E402

  `worker_init_fn` and `make_generator` are **not** imported at either state.

### Inside `src/seeding.py`, identical at both states

Each receives `seed` unmodified. No derived value and no constant is used.

| line | call | source seeded |
| --- | --- | --- |
| `:26` | `os.environ["PYTHONHASHSEED"] = str(seed)` | hash seed |
| `:27` | `random.seed(seed)` | python `random` |
| `:28` | `np.random.seed(seed)` | numpy global |
| `:31` | `torch.manual_seed(seed)` | torch CPU |
| `:33` | `torch.cuda.manual_seed_all(seed)` | torch CUDA |

`:31` onward sits inside `if torch is not None:` at `:30`. `:33` sits inside
`if torch.cuda.is_available():` at `:32`.

### Every `torch.Generator` construction

- `src/seeding.py:72-73`, both states, inside `make_generator`:

      g = torch.Generator()
      g.manual_seed(int(seed))

  Receives the seed passed to `make_generator`. **Not called from
  `tools/train_arm.py` at either state**, which imports only
  `seed_everything`.

- **A only**, `src/tubelets.py:111`:

      g = torch.Generator()

  **No `manual_seed` is called on this object.** It receives no value of any
  kind. This construction does not exist at B.

## 2. What consumes each seeded source

| consumer | state A | state B |
| --- | --- | --- |
| head initialisation | `tools/epoch_timing.py:120, 125, 130, 139` | `src/arms.py:160, 165, 170, 179` |
| shuffle | `tools/train_arm.py:131` | `tools/train_arm.py:133` |
| flip draw | `src/tubelets.py:113` | `src/tubelets.py:115` |
| brightness and contrast jitter | `src/tubelets.py:115-116` | `src/tubelets.py:117-118` |
| sampler | none constructed | none constructed |

### Head initialisation

The replaced classification head consumes the torch CPU generator seeded at
`src/seeding.py:31`. A builds through `from epoch_timing import build_arm`;
B through `from src.arms import build_arm`.

- A `tools/epoch_timing.py:119-120`

      m = resnet18(weights=None)
      m.fc = nn.Linear(m.fc.in_features, N_CLASSES)

  with the same `nn.Linear` replacement at `:125`, `:130` and `:139`.

- B `tools/train_arm.py:176-177`

      pre = args.weights == "pretrained"
      model, _ = build_arm(args.arm, T, verify=pre, pretrained=pre)

  and `src/arms.py:155-156`

      if not pretrained:
          weights = None

  with the `nn.Linear` replacement at `src/arms.py:160`, `:165`, `:170` and
  `:179`.

### Flip draw, the difference between the states

- A `src/tubelets.py:111-113`

      g = torch.Generator()
      # one draw per tubelet, all frames share it
      if torch.rand(1, generator=g).item() < self.hflip_p:

- B `src/tubelets.py:111-115`

      # One draw per tubelet, all frames share it. NOT via a fresh
      # torch.Generator(): that carries a fixed default seed, so
      # torch.rand through it returns the same value every call and the
      # flip becomes a constant rather than a coin toss.
      if np.random.random() < self.hflip_p:

### Jitter draws

Identical text at both states, consuming the numpy global generator seeded at
`src/seeding.py:28`. A `src/tubelets.py:115-116`, B `src/tubelets.py:117-118`:

    b = np.random.uniform(*self.brightness)
    c = np.random.uniform(*self.contrast)

### Sampler

No explicit sampler is constructed at either state. `shuffle=True` is the only
specification; see section 3.

### Randomness with no seeding found

- **DataLoader worker processes, both states.** `worker_init_fn` is defined at
  `src/seeding.py:57-65`:

      def worker_init_fn(worker_id: int) -> None:
          """DataLoader worker seeding. Pass as worker_init_fn to every DataLoader."""
          ...
          random.seed(base)
          np.random.seed(base)

  It is not passed to the DataLoader at either state, and is not imported by
  `tools/train_arm.py` at either state. `num_workers` defaults to 4, so the
  numpy draws above run in worker processes whose numpy state is not set by
  this function.

- **A `src/tubelets.py:111`.** The generator there receives no seed value.

- `src/vibe.py:98` holds a separately seeded generator,
  `self.rng = np.random.default_rng(self.seed)`, consumed at `:162` and
  `:169`. It is not reached from `tools/train_arm.py` at either state.

## 3. Dataloader

Each state constructs exactly one DataLoader, verified by counting
`DataLoader(` occurrences in `tools/train_arm.py` at each commit.

- A `tools/train_arm.py:130-132`

      loader = torch.utils.data.DataLoader(
          train_ds, batch_size=batch, shuffle=True, num_workers=args.workers,
          pin_memory=True, drop_last=False)

- B `tools/train_arm.py:132-134`: identical text.

| question | A | B |
| --- | --- | --- |
| `num_workers` | `args.workers`, default **4** | `args.workers`, default **4** |
| `worker_init_fn` passed | **no** | **no** |
| `generator` passed | **no** | **no** |
| shuffle via a seeded generator | **no explicit generator** | **no explicit generator** |

The default is declared at `tools/train_arm.py:91` in **both** states:

    ap.add_argument("--workers", type=int, default=4)

The string `generator=` does not occur in `tools/train_arm.py` at either
state.

## 4. Determinism flags

All in `src/seeding.py`, identical at both states. Reached with
`deterministic=True` and `cudnn_benchmark=False`, those being the second and
third arguments of the `seed_everything(args.seed, True, False)` call quoted
in section 1.

| line | statement | resulting value |
| --- | --- | --- |
| `:34` | `torch.backends.cudnn.benchmark = bool(cudnn_benchmark)` | `False` |
| `:35` | `torch.backends.cudnn.deterministic = bool(deterministic)` | `True` |
| `:37` | `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` | set if absent |
| `:39` | `torch.use_deterministic_algorithms(True, warn_only=True)` | enabled |
| `:41` | `torch.use_deterministic_algorithms(True)` | the `except TypeError` fallback |

`:34` and `:35` sit inside `if torch is not None:` at `:30`. Lines `:37`,
`:39` and `:41` additionally sit inside `if deterministic:` at `:36`.

None of these four settings is absent at either state.

## 5. How the run selects its negative list

The chain is identical at both states.

1. The seed is passed as the third positional argument to the dataset.
   A `tools/train_arm.py:124-125`, B `tools/train_arm.py:126-127`:

       train_ds = TubeletDataset(
           args.store, args.manifest, args.seed, args.arm, mean, std,

2. `src/tubelets.py:89`, both states:

       self.rows = rows_for_seed(read_manifest(manifest), seed)

3. `src/tubelets.py:54-63`, both states, the link itself:

       def rows_for_seed(manifest: list[dict], seed: int) -> list[tuple[int, int]]:
           """Return (store_index, label) for one run. Positives are in every run;
           a negative is in this run only if its seeds field names this seed."""
           out = []
           for r in manifest:
               if r["label"] == "positive":
                   out.append((int(r["index"]), 1))
               elif str(seed) in (r["seeds"] or "").split("|"):
                   out.append((int(r["index"]), 0))
           return out

4. The manifest path defaults at `tools/train_arm.py:86`:

       ap.add_argument("--manifest", default="data/tubelets/manifest.csv")

5. The `seeds` column is written by `tools/materialise_tubelets.py`, identical
   at both states:

   - `:93` `ap.add_argument("--seeds", default="1,2,3")`
   - `:128` `for seed in seeds:`
   - `:129` `p = os.path.join(args.negatives, f"negatives_seed{seed}.csv")`
   - `:138` `add(row, jx, "negative", seed)`
   - `:117` `e["seeds"].add(seed)`
   - `:203` `"|".join(str(s) for s in sorted(e["seeds"]))`

## Differences between the two states, stated explicitly

1. **Flip draw**, `src/tubelets.py`. A draws through
   `torch.rand(1, generator=g)` where `g = torch.Generator()` carries no
   `manual_seed`; B draws through `np.random.random()`. This is the only
   change to how randomness is consumed.
2. **`build_arm` source and weights.** A imports from `epoch_timing` and
   constructs backbones with `weights=None` (`tools/epoch_timing.py:119`,
   `:124`, `:129`, `:138`). B imports from `src.arms`, adds `--weights` at
   `tools/train_arm.py:92-93` with `choices=["pretrained", "scratch"]`, and
   calls `build_arm(args.arm, T, verify=pre, pretrained=pre)` at `:177`.
3. **Line offsets.** B is shifted by two lines from `tools/train_arm.py:92`
   onward.
4. **Not randomness.** B passes `batch=64` to `streaming_eval.evaluate` at
   `tools/train_arm.py:214` and `:247`.

Sections 1, 3, 4 and 5 are otherwise identical across the two states, as is
the whole of `src/seeding.py`.
