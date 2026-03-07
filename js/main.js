// Smooth scroll for nav links — offset for sticky nav height
document.querySelectorAll('#site-nav a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            e.preventDefault();
            const nav = document.getElementById('site-nav');
            const offset = nav ? nav.offsetHeight + 8 : 0;
            const top = target.getBoundingClientRect().top + window.scrollY - offset;
            window.scrollTo({ top: top, behavior: 'smooth' });
        }
    });
});