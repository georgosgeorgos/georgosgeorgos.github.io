document.addEventListener("DOMContentLoaded", function() {
    if (typeof katex === "undefined") return;

    // Render inline math: <span class="math inline">...</span>
    document.querySelectorAll(".math.inline").forEach(function(el) {
        katex.render(el.textContent, el, { throwOnError: false, displayMode: false });
    });

    // Render display math: <span class="math display">...</span>
    document.querySelectorAll(".math.display").forEach(function(el) {
        katex.render(el.textContent, el, { throwOnError: false, displayMode: true });
    });
});
