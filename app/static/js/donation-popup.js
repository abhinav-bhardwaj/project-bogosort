class DonationPopup {
  constructor(options = {}) {
    this.options = {
      showDelay: 1500,
      storageKey: 'donationPopupDismissed',
      ...options
    };
    this.init();
  }

  init() {
    // Only show once per browser session (sessionStorage resets on tab/window close)
    if (sessionStorage.getItem(this.options.storageKey)) return;

    this.createPopup();
    this.attachEventListeners();
    setTimeout(() => this.showPopup(), this.options.showDelay);
  }

  createPopup() {
    const el = document.createElement('div');
    el.className = 'donation-modal-overlay';
    el.id = 'donationOverlay';
    el.setAttribute('aria-modal', 'true');
    el.setAttribute('role', 'dialog');
    el.setAttribute('aria-labelledby', 'donationTitle');
    el.innerHTML = `
      <div class="donation-modal">
        <div class="donation-modal-header">
          <h2 id="donationTitle">Support Open Science</h2>
          <button class="donation-modal-close" aria-label="Close">&times;</button>
        </div>
        <div class="donation-modal-body">
          <p class="donation-lead">
            Help us advance toxicity detection and content moderation research.
          </p>
          <div class="donation-benefits">
            <div class="benefit-item">
              <span class="benefit-icon"></span>
              <span class="benefit-text">Support cutting-edge ML research</span>
            </div>
            <div class="benefit-item">
              <span class="benefit-icon"></span>
              <span class="benefit-text">Enable open science for all</span>
            </div>
            <div class="benefit-item">
              <span class="benefit-icon"></span>
              <span class="benefit-text">Join a global research community</span>
            </div>
          </div>
          <p class="donation-description">
            Your contribution helps us develop better toxicity detection models and make them freely
            available to researchers worldwide. 100% of proceeds fund research infrastructure and
            model development.
          </p>
        </div>
        <div class="donation-modal-footer">
          <button class="donation-btn donation-btn-secondary" id="donationNotNow">Not now</button>
          <button class="donation-btn donation-btn-primary" id="donationContribute">Contribute to Research</button>
        </div>
      </div>`;
    document.body.appendChild(el);
  }

  attachEventListeners() {
    const overlay     = document.getElementById('donationOverlay');
    const closeBtn    = overlay?.querySelector('.donation-modal-close');
    const notNowBtn   = document.getElementById('donationNotNow');
    const contributeBtn = document.getElementById('donationContribute');

    closeBtn?.addEventListener('click',    () => this.dismiss());
    notNowBtn?.addEventListener('click',   () => this.dismiss());
    contributeBtn?.addEventListener('click', () => {
      window.location.href = '#donate';
      this.dismiss();
    });

    // Click outside modal to dismiss
    overlay?.addEventListener('click', e => {
      if (e.target === overlay) this.dismiss();
    });

    // Escape key
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') this.dismiss();
    }, { once: true });
  }

  showPopup() {
    const overlay = document.getElementById('donationOverlay');
    if (!overlay) return;
    // Trigger CSS transition: adding the class goes from opacity:0 to opacity:1
    // The overlay is already in the DOM (display:flex via CSS class), just invisible
    requestAnimationFrame(() => {
      overlay.classList.add('donation-modal-show');
    });
  }

  dismiss() {
    const overlay = document.getElementById('donationOverlay');
    if (!overlay) return;

    overlay.classList.remove('donation-modal-show');
    overlay.addEventListener('transitionend', () => overlay.remove(), { once: true });

    // Mark dismissed for this session so navigating pages doesn't re-trigger
    sessionStorage.setItem(this.options.storageKey, '1');
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new DonationPopup());
} else {
  new DonationPopup();
}
