// Smooth scroll for nav links
document.querySelectorAll('#site-nav a').forEach(link => {
    link.addEventListener('click', e => {
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Collapsible abstracts
// Finds the last <p> in each right-column cell (75% width) and hides it with a toggle
document.querySelectorAll('td[width="75%"]').forEach(td => {
    const paragraphs = td.querySelectorAll('p');
    if (paragraphs.length < 2) return;

    const abstract = paragraphs[paragraphs.length - 1];
    abstract.classList.add('abstract');

    const toggle = document.createElement('span');
    toggle.className = 'abstract-toggle';
    toggle.textContent = '▸ abstract';
    toggle.addEventListener('click', () => {
        const open = abstract.style.display === 'block';
        abstract.style.display = open ? 'none' : 'block';
        toggle.textContent = open ? '▸ abstract' : '▾ abstract';
    });

    abstract.parentNode.insertBefore(toggle, abstract);
});
