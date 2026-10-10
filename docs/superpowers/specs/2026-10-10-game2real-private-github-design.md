# Game2Real-Bench Private GitHub Release Design

## Objective

Prepare a polished, reproducible, private GitHub preview of the Game2Real-Bench code release. The repository should be ready to switch to public visibility after the paper and licensing decisions are finalized, without requiring a structural rewrite.

## Reference Direction

The presentation combines the concise benchmark narrative used by VISTA-Bench with the stronger navigation and reproducibility patterns used by mature repositories such as MathVista and MMMU:

- a clear research question and link row at the top;
- one primary overview figure before implementation details;
- an at-a-glance benchmark summary;
- copyable installation, download, evaluation, and validation commands;
- separate documentation for data fields and scoring behavior;
- licensing and attribution boundaries stated close to the relevant assets;
- CI-backed tests and a release validation command.

## Approved Visual Direction

Use the approved "Paper-ready Hybrid" layout.

1. Centered title and full paper subtitle.
2. Compact badges and repository links.
3. The paper construction pipeline as the primary overview figure.
4. The capability-aligned Toy Game and Real World task figure in the benchmark-design section.
5. Four compact benchmark statistics: 1,400 evaluation samples, 18 toy-game tasks, four capabilities, and two aligned settings.
6. No unpublished result plots in the initial README. Correlation, in-context learning, thinking, and failure-case figures remain outside the GitHub release until explicitly approved.

The selected paper figures are converted from their vector PDFs to sharp, web-friendly PNG files. README-specific cropping may remove unused paper-layout whitespace but must not alter scientific content.

## README Information Architecture

1. Title, subtitle, badges, and links.
2. Overview and construction-pipeline figure.
3. Benchmark at a glance.
4. Capability-aligned benchmark design and task figure.
5. Repository layout.
6. Quick start:
   - install;
   - download the private Hugging Face dataset;
   - prepare predictions;
   - run deterministic evaluation.
7. Evaluation details and VLMEvalKit adapter.
8. Release validation and test commands.
9. Data/licensing boundary.
10. Citation placeholder for the unpublished manuscript.

## Repository Structure

```text
.
├── .github/workflows/ci.yml
├── assets/figures/
├── docs/
│   ├── DATA_FORMAT.md
│   ├── EVALUATION.md
│   └── superpowers/specs/
├── examples/
├── game2real/
├── scripts/
├── tests/
├── vlmevalkit_adapter/
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── THIRD_PARTY_LICENSES.md
└── pyproject.toml
```

The data remain in the separate private Hugging Face dataset repository. No dataset images or source-only generator material are copied into the GitHub code repository.

## Functional Scope

- Portable TSV loading and image-path resolution.
- Deterministic Toy Game and Real World scoring.
- Support for all released Real World evaluator types.
- Hugging Face snapshot download helper.
- Sanitized dataset-release builder.
- VLMEvalKit path adapter.
- Publication validator and unit tests.
- Private-by-default GitHub publication script.

## Privacy and Release Safety

- Create `QingAnLiu/Game2Real-Bench` as a private repository.
- Do not include credentials, local absolute paths, paper source files, author-only notes, experiment logs, or unpublished numeric result plots.
- Keep the Hugging Face dataset link, while noting that access follows the dataset repository visibility.
- Default all scripted GitHub creation to private visibility unless an explicit public flag is supplied in a future revision.

## Verification

Before upload:

1. run the full unit-test suite;
2. run the release validator against the prepared dataset and code roots;
3. scan tracked content for private paths, secrets, caches, and generated temporary files;
4. confirm README image rendering and all relative links;
5. initialize a clean Git repository and review the exact tracked-file list;
6. create and push the private GitHub repository;
7. verify remote visibility and README rendering.

