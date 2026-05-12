(function () {
  function initOutlineNav(navId) {
    const nav = document.getElementById(navId);
    if (!nav) return;

    const links = Array.from(nav.querySelectorAll('a[href^="#"]'));
    const targets = links
      .map(a => ({ link: a, el: document.querySelector(a.getAttribute('href')) }))
      .filter(t => t.el);

    if (!targets.length) return;

    function setActive(link) {
      links.forEach(a => a.classList.remove('active'));
      link.classList.add('active');
      link.scrollIntoView({ block: 'nearest' });
    }

    // Click: smooth scroll + immediately mark active
    links.forEach(a => {
      a.addEventListener('click', e => {
        e.preventDefault();
        const target = document.querySelector(a.getAttribute('href'));
        if (!target) return;
        setActive(a);
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    });

    // Scroll-spy via IntersectionObserver
    if (!('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const t = targets.find(t => t.el === entry.target);
          if (t) setActive(t.link);
        }
      });
    }, {
      rootMargin: '-5% 0px -80% 0px',
      threshold: 0
    });

    targets.forEach(t => observer.observe(t.el));

    // Activate first item on load
    setActive(targets[0].link);
  }

  document.addEventListener('DOMContentLoaded', () => {
    initOutlineNav('nerdyOutline');
    initOutlineNav('edaOutline');
  });
})();
