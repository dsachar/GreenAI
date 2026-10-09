// Zero-build client-side i18n engine for Green AI
(function() {
  const STORAGE_KEY = "greenai_lang";
  let currentLang = "en";

  // Determine initial language
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "fr" || saved === "en") {
      currentLang = saved;
    } else if (navigator.language && navigator.language.toLowerCase().startsWith("fr")) {
      currentLang = "fr";
    }
  } catch (e) {
    // Local storage unavailable (sandboxed / privacy mode)
  }

  function getNested(obj, path) {
    if (!obj || typeof obj !== "object") return undefined;
    const parts = path.split(".");
    let cur = obj;
    for (const p of parts) {
      if (cur == null || typeof cur !== "object") return undefined;
      cur = cur[p];
    }
    return cur;
  }

  function t(key, vars = {}) {
    const locales = window.LOCALES || {};
    const dict = locales[currentLang] || {};
    const fallback = locales["en"] || {};

    let val = getNested(dict, key);
    if (val === undefined) {
      val = getNested(fallback, key);
    }
    if (val === undefined) {
      return key;
    }

    if (typeof val === "string" && Object.keys(vars).length > 0) {
      for (const [k, v] of Object.entries(vars)) {
        val = val.replaceAll(`{${k}}`, v);
      }
    }
    return val;
  }

  function getLang() {
    return currentLang;
  }

  function setLanguage(lang) {
    if (lang !== "en" && lang !== "fr") return;
    currentLang = lang;
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch (e) {}

    translateDom();
    window.dispatchEvent(new CustomEvent("languagechange", { detail: { lang } }));
  }

  function translateDom() {
    document.documentElement.lang = currentLang;

    // textContent
    document.querySelectorAll("[data-i18n]").forEach(el => {
      const key = el.getAttribute("data-i18n");
      const text = t(key);
      if (text && text !== key) {
        el.textContent = text;
      }
    });

    // innerHTML (for elements with rich markup)
    document.querySelectorAll("[data-i18n-html]").forEach(el => {
      const key = el.getAttribute("data-i18n-html");
      const html = t(key);
      if (html && html !== key) {
        el.innerHTML = html;
      }
    });

    // aria-label
    document.querySelectorAll("[data-i18n-aria]").forEach(el => {
      const key = el.getAttribute("data-i18n-aria");
      const label = t(key);
      if (label && label !== key) {
        el.setAttribute("aria-label", label);
      }
    });

    // title
    document.querySelectorAll("[data-i18n-title]").forEach(el => {
      const key = el.getAttribute("data-i18n-title");
      const title = t(key);
      if (title && title !== key) {
        el.setAttribute("title", title);
      }
    });

    // Language switcher buttons
    document.querySelectorAll(".lang-btn").forEach(btn => {
      const isActive = btn.dataset.lang === currentLang;
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-pressed", isActive ? "true" : "false");
    });
  }

  // Locale-aware formatting helpers
  function fmtNum(v) {
    if (typeof v !== "number" || isNaN(v)) return v;
    return Math.round(v).toLocaleString(currentLang === "fr" ? "fr-FR" : "en-US");
  }

  function fmtPct(v, decimals = 1) {
    if (typeof v !== "number" || isNaN(v)) return v;
    const str = currentLang === "fr"
      ? v.toFixed(decimals).replace(".", ",") + " %"
      : v.toFixed(decimals) + "%";
    return str;
  }

  function fmtMonths(m) {
    if (currentLang === "fr") {
      if (m < 12) return `${m} mois`;
      const y = +(m / 12).toFixed(1);
      const yStr = String(y).replace(".", ",");
      return `${m} mois (${yStr} an${y >= 2 ? "s" : ""})`;
    } else {
      if (m < 12) return `${m} month${m === 1 ? "" : "s"}`;
      const y = +(m / 12).toFixed(1);
      return `${m} months (${y} yr${y === 1 ? "" : "s"})`;
    }
  }

  // Expose i18n globally
  window.i18n = {
    t,
    getLang,
    setLanguage,
    translateDom,
    fmtNum,
    fmtPct,
    fmtMonths
  };

  // Wire up switcher click handlers on document ready
  document.addEventListener("DOMContentLoaded", () => {
    translateDom();
    document.addEventListener("click", ev => {
      const btn = ev.target.closest(".lang-btn");
      if (btn && btn.dataset.lang) {
        setLanguage(btn.dataset.lang);
      }
    });
  });
})();
