document.addEventListener("DOMContentLoaded", function() {
    // --- Auto-generate TOC from h2/h3 headings ---
    var tocNav = document.getElementById("toc");
    var headings = document.querySelectorAll(".essay-container h2, .essay-container h3");
    if (tocNav && headings.length > 0) {
        var ul = document.createElement("ul");
        headings.forEach(function(h) {
            if (!h.id) return;
            var li = document.createElement("li");
            li.className = h.tagName === "H3" ? "toc-h3" : "toc-h2";
            var a = document.createElement("a");
            a.href = "#" + h.id;
            a.textContent = h.textContent;
            li.appendChild(a);
            ul.appendChild(li);
        });
        tocNav.appendChild(ul);
    }

    // --- Smooth scroll TOC links ---
    tocNav && tocNav.addEventListener("click", function(e) {
        var link = e.target.closest("a");
        if (!link) return;
        var target = document.querySelector(link.getAttribute("href"));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: "smooth" });
        }
    });

    // --- Highlight current TOC entry on scroll ---
    var tocLinks = tocNav ? tocNav.querySelectorAll("a") : [];
    if (tocLinks.length > 0) {
        var onScroll = function() {
            var scrollPos = window.scrollY + 80;
            var current = null;
            headings.forEach(function(h) {
                if (h.offsetTop <= scrollPos) current = h.id;
            });
            tocLinks.forEach(function(a) {
                a.classList.toggle("active", a.getAttribute("href") === "#" + current);
            });
        };
        window.addEventListener("scroll", onScroll, { passive: true });
        onScroll();
    }

    // --- Position sidenotes in right column ---
    // In markdown, write sidenotes as: <span class="sidenote">Your note here.</span>
    var sidenoteColumn = document.querySelector(".essay-sidenotes");
    var sidenotes = document.querySelectorAll(".essay-container .sidenote");
    if (sidenoteColumn && sidenotes.length > 0) {
        var counter = 0;
        sidenotes.forEach(function(note) {
            counter++;
            // Insert superscript number in the text
            var sup = document.createElement("sup");
            sup.className = "sidenote-ref";
            sup.textContent = counter;
            note.parentNode.insertBefore(sup, note);

            // Create the margin note
            var marginNote = document.createElement("div");
            marginNote.className = "sidenote-content";
            marginNote.innerHTML = "<sup>" + counter + "</sup> " + note.innerHTML;

            // Position aligned with the reference
            var refTop = sup.offsetTop;
            marginNote.style.position = "absolute";
            marginNote.style.top = refTop + "px";

            sidenoteColumn.appendChild(marginNote);

            // Hide the inline note on wide screens (CSS handles this)
            note.classList.add("sidenote-inline");
        });
    }
});
