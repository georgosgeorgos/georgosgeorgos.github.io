# Essays

Pandoc-based essay system: write Markdown, convert to clean static HTML.

## Structure

```
essays/
  index.html              # listing page (all essays, date + title)
  template.html           # pandoc HTML template (nav, footer, KaTeX, medium-zoom)
  build.sh                # shell script wrapping pandoc command
  css/essay.css           # essay-specific styles
  js/essay.js             # TOC generation + sidenotes
  katex/                  # self-hosted KaTeX (CSS, JS, fonts)
  quotes.json             # epigraph quotes for listing page
  <essay-folder>/
    essay.md              # source Markdown with YAML front matter
    index.html            # pandoc output (generated)
    images/               # essay-specific assets
```

## Workflow

1. Create folder: `essays/my-new-essay/`
2. Write `essay.md` with YAML front matter (title, date, author)
3. Run `./build.sh my-new-essay`
4. Add entry to `essays/index.html` listing
5. Commit and push

## Prerequisites

- [pandoc](https://pandoc.org/) (`brew install pandoc`)

## Setup

After cloning, run once to enable the pre-commit hook:

```bash
git config core.hooksPath hooks/
```

This configures git to use the tracked `hooks/` directory. The pre-commit hook automatically rebuilds any changed essay HTML (via `build.sh`) and stages the result before each commit.

## Features

- Self-hosted KaTeX for math rendering (inline `$...$` and display `$$...$$`)
- Auto-generated table of contents (via JS)
- Sidenotes support
- medium-zoom for image click-to-zoom
- Syntax-highlighted code blocks
- Responsive layout
- Random epigraph quotes on listing page

## Future Improvements

- Dark mode support (`prefers-color-scheme: dark`)
- Reading time estimate on listing page and essay header
- Prev/Next navigation between essays
- Pandoc footnote support (`[^1]` syntax)
- Mobile TOC (collapsible instead of hidden)
- Code copy button on code blocks
