# TODO: Add Essays Section (thinkingmachines.ai style)

Pandoc-based workflow: write Markdown, convert to clean static HTML.
Reference: https://thinkingmachines.ai/blog/ and https://thinkingmachines.ai/blog/on-policy-distillation/

## Prerequisites

- [x] Install pandoc: `brew install pandoc`

## 1. Directory structure

Create `essays/` at repo root:
```
essays/
  index.html              # listing page (all essays, date + title + author)
  css/
    essay.css             # essay-specific styles (post page + listing page)
  katex/
    katex.min.css         # self-hosted KaTeX
    katex.min.js
    contrib/
      auto-render.min.js
    fonts/                # KaTeX font files
  template.html           # pandoc HTML template (nav, footer, KaTeX, medium-zoom)
  build.sh                # shell script wrapping pandoc command
  my-first-essay/
    essay.md              # source Markdown (not served, or gitignored)
    index.html            # pandoc output
    images/               # essay-specific assets (SVGs, PNGs)
```

## 2. Pandoc HTML template (`essays/template.html`)

- Single-column layout, max-width ~720px, centered
- Header: link back to main site + essay listing
- Metadata block: title (large), author, date (from pandoc front matter)
- Body: rendered from Markdown
- Footer: prev/next navigation (arrow keys), back to listing
- Includes:
  - Self-hosted KaTeX (lazy-loaded when math detected)
  - medium-zoom (CDN or self-hosted) for image click-to-zoom
  - Pandoc's built-in syntax highlighting CSS

## 3. Build script (`essays/build.sh`)

Simple wrapper around pandoc:
```bash
pandoc essay.md -o index.html \
  --template=../template.html \
  --katex \
  --highlight-style=kate \
  --metadata title="My Essay" \
  --metadata date="2026-03-06" \
  --metadata author="Giorgio Giannone"
```
Or read title/date/author from YAML front matter in the .md file.

## 4. Self-host KaTeX

- Download latest KaTeX release (CSS, JS, auto-render, fonts)
- Place in `essays/katex/`
- Lazy-load: only init KaTeX when page contains `$$`, `$`, `\(`, or `\[`
- Support both inline and display math

## 5. Essay listing page (`essays/index.html`)

- Minimal vertical list like thinkingmachines.ai/blog/
- Each entry: date, title, author (linked)
- Sorted newest first
- Manually maintained (no build step generates this)
- Style consistent with main site (same nav font, link color `#4a6741`)

## 6. CSS styling (`essays/css/essay.css`)

- Sans-serif font stack (match main site: `system-ui, -apple-system, sans-serif`)
- Body: good line-height (~1.6-1.7), readable font size (16-18px)
- Headings: clean weight, good spacing
- Code blocks: light background, border-radius, monospace font
- Blockquotes: left border accent
- Tables: clean borders
- Images: optional captions, medium-zoom on click
- Responsive: readable on mobile
- Print-friendly (optional)

## 7. Main site integration

- Add "Essays" link to `#site-nav` in `index.html`
- Link points to `essays/`

## 8. Sample essay

- Create one sample post demonstrating all features:
  - YAML front matter (title, date, author)
  - Inline math: `$x^2$`
  - Display math: `$$\nabla_\theta \mathcal{L}$$`
  - Code block with syntax highlighting
  - An image with caption
  - A blockquote
  - A table
- Run `build.sh` to generate index.html

## Workflow

1. Create folder: `essays/my-new-essay/`
2. Write `essay.md` with YAML front matter
3. Run `build.sh` (or `pandoc` directly)
4. Add entry to `essays/index.html` listing
5. Commit and push

## Notes

- Each essay is a folder (`essays/my-essay/`) with source .md and output index.html
- Assets (images, SVGs) live alongside the essay
- No framework, no dependencies beyond pandoc
- GitHub Pages serves the HTML as-is
- The old `blog/` Jekyll folder can be removed later
