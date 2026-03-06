# CLAUDE.md

## Project Overview

Personal academic website for Giorgio Giannone — static HTML/CSS site hosted on GitHub Pages at `georgosgeorgos.github.io`.

## Structure

- `index.html` — Main single-page site (About, Research, Publications, Projects, Open-source, Datasets, Teaching, Patents, Theses)
- `css/style.css` — Main stylesheet
- `blog/` — Jekyll-based blog ("georgos blog")
  - `_config.yml` — Jekyll config (kramdown + rouge)
  - `_layouts/` — default, page, post layouts
  - `_includes/` — header, footer, head partials
  - `_posts/` — Blog posts in markdown
- `assets/` — Images, logos, CV PDF

## Tech Stack

- Static HTML + CSS (no build system for main site)
- Jekyll for the blog subsection
- GitHub Pages for hosting
- Font: Ubuntu Mono (via cdnfonts)

## Branches

- `master` — Main/production branch (deployed via GitHub Pages)
- `redesign/modernize` — Active redesign branch

## Conventions

- The main page is a single large `index.html` with table-based layout
- Navigation uses a sticky nav bar with anchor links to page sections
- Publications are organized as "Selected" and "All" with paper images, links, and metadata
- No JavaScript framework — vanilla HTML/CSS
