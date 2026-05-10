class DonationPopup {
  constructor(options = {}) {
    this.options = {
      showOnPageChange: true,
      showDelay: 500,
      storageKey: 'donationPopupDismissed',
      dismissDuration: 3600000,
      ...options
    };
    
    this.currentPage = this.getCurrentPage();
    this.dismissed = this.isDismissed();
    this.init();
  }

  init() {
    this.createPopup();
    this.attachEventListeners();
    this.observePageChanges();
    
    // Show popup on initial page load if not dismissed
    if (!this.dismissed) {
      setTimeout(() => this.showPopup(), this.options.showDelay);
    }
  }

  getCurrentPage() {
    return window.location.pathname;
  }

  isDismissed() {
    const stored = localStorage.getItem(this.options.storageKey);
    if (!stored) return false;
    
    try {
      const data = JSON.parse(stored);
      const now = Date.now();
      return now - data.timestamp < this.options.dismissDuration;
    } catch {
      return false;
    }
  }

  createPopup() {
    const popupHTML = `
      <div class="donation-modal-overlay" id="donationOverlay" style="display: none;">
        <div class="donation-modal">
          <div class="donation-modal-header">
            <h2>Support Open Science</h2>
            <button class="donation-modal-close" aria-label="Close donation popup">&times;</button>
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
              Your contribution helps us develop better toxicity detection models and make them freely available to researchers worldwide. 100% of proceeds fund research infrastructure and model development.
            </p>
          </div>
          
          <div class="donation-modal-footer">
            <button class="donation-btn donation-btn-secondary" id="donationNotNow">
              Not now
            </button>
            <button class="donation-btn donation-btn-primary" id="donationContribute">
              Contribute to Research
            </button>
          </div>
        </div>
      </div>
    `;

    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = popupHTML;
    document.body.appendChild(tempDiv.firstElementChild);
  }

  attachEventListeners() {
    const overlay = document.getElementById('donationOverlay');
    const closeBtn = document.querySelector('.donation-modal-close');
    const notNowBtn = document.getElementById('donationNotNow');
    const contributeBtn = document.getElementById('donationContribute');

    if (closeBtn) {
      closeBtn.addEventListener('click', () => this.dismissPopup());
    }

    if (notNowBtn) {
      notNowBtn.addEventListener('click', () => this.dismissPopup());
    }

    if (contributeBtn) {
      contributeBtn.addEventListener('click', () => {
        window.location.href = '#donate';
        this.dismissPopup();
      });
    }

    if (overlay) {
      overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
          this.dismissPopup();
        }
      });
    }
  }

  observePageChanges() {
    let lastPage = this.currentPage;

    const checkPageChange = () => {
      const newPage = this.getCurrentPage();
      
      if (newPage !== lastPage) {
        lastPage = newPage;
        this.currentPage = newPage;
        
        if (this.options.showOnPageChange && !this.dismissed) {
          this.showPopup();
        }
      }
    };

    window.addEventListener('popstate', checkPageChange);

    const originalPushState = window.history.pushState;
    const originalReplaceState = window.history.replaceState;

    window.history.pushState = function(...args) {
      originalPushState.apply(this, args);
      checkPageChange();
    };

    window.history.replaceState = function(...args) {
      originalReplaceState.apply(this, args);
      checkPageChange();
    };

    const originalFetch = window.fetch;
    window.fetch = function(...args) {
      return originalFetch.apply(this, args).then(response => {
        checkPageChange();
        return response;
      });
    };
  }

  showPopup() {
    const overlay = document.getElementById('donationOverlay');
    if (overlay) {
      overlay.style.display = 'flex';
      setTimeout(() => {
        overlay.classList.add('donation-modal-show');
      }, 50);
    }
  }

  dismissPopup() {
    const overlay = document.getElementById('donationOverlay');
    if (overlay) {
      overlay.classList.remove('donation-modal-show');
      setTimeout(() => {
        overlay.style.display = 'none';
      }, 300);
    }

    const now = Date.now();
    localStorage.setItem(
      this.options.storageKey,
      JSON.stringify({ timestamp: now })
    );
    this.dismissed = true;
  }

  reset() {
    localStorage.removeItem(this.options.storageKey);
    this.dismissed = false;
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.donationPopup = new DonationPopup({
      showOnPageChange: true,
      showDelay: 800,
      dismissDuration: 3600000
    });
  });
} else {
  window.donationPopup = new DonationPopup({
    showOnPageChange: true,
    showDelay: 800,
    dismissDuration: 3600000
  });
}