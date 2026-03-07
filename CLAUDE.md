# CLAUDE.md

## Project Overview

Personal academic website for Giorgio Giannone — static HTML/CSS/JS site hosted on GitHub Pages at `georgosgeorgos.github.io`.

## Structure

- `index.html` — Main single-page site (About, Research, Publications, Projects, Open-source, Datasets, Teaching, Patents, Theses)
- `css/style.css` — Main stylesheet
- `js/main.js` — Smooth scroll and minor interactions
- `blog/` — Jekyll-based blog ("georgos blog")
  - `_config.yml` — Jekyll config (kramdown + rouge)
  - `_layouts/` — default, page, post layouts
  - `_includes/` — header, footer, head partials
  - `_posts/` — Blog posts in markdown
- `writing/` — Standalone writing with KaTeX math support
  - `template.html` + `build.sh` — Essay templating
  - `quotes.jsonl` — Epigraph quotes
  - `essays/`, `notes/`, `predictions/` — Content categories
- `assets/` — Images, logos, CV PDF

## Tech Stack

- Static HTML + CSS + vanilla JS (no build system for main site)
- CSS Grid layout with mobile responsiveness
- Jekyll for the blog subsection
- GitHub Pages for hosting
- Font: Ubuntu Mono (via cdnfonts)
- KaTeX for math rendering in essays

## Branches

- `master` — Production branch (deployed via GitHub Pages)
- `redesign/modernize` — Active redesign branch

## Conventions

- The main page is a single `index.html` using CSS Grid for layout
- Navigation uses a sticky nav bar with anchor links to page sections
- Publications are organized as "Selected" and "All" with paper images, links, and metadata
- No JavaScript frameworks — vanilla HTML/CSS/JS only
- Keep changes minimal and avoid over-engineering
