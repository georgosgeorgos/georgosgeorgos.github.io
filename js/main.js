// Smooth scroll for nav links — offset for sticky nav height
document.querySelectorAll('#site-nav a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            e.preventDefault();
            const details = target.closest('details');
            if (details && !details.open) details.open = true;
            const nav = document.getElementById('site-nav');
            const offset = nav ? nav.offsetHeight + 8 : 0;
            const top = target.getBoundingClientRect().top + window.scrollY - offset;
            window.scrollTo({ top: top, behavior: 'smooth' });
        }
    });
});

// Publication filter: selected / all
const pubSection = document.getElementById('publications').closest('details');
const filterOptions = document.querySelectorAll('.pub-filter-option');

function filterPublications(filter) {
    const rows = pubSection.querySelectorAll('.entry-row');
    rows.forEach(row => {
        if (filter === 'selected' && !row.classList.contains('paper-selected')) {
            row.style.display = 'none';
        } else {
            row.style.display = '';
        }
    });
    filterOptions.forEach(opt => {
        opt.classList.toggle('active', opt.dataset.filter === filter);
    });
}

filterOptions.forEach(opt => {
    opt.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        if (!pubSection.open) pubSection.open = true;
        filterPublications(opt.dataset.filter);
    });
});

// Default: show only selected papers
filterPublications('selected');