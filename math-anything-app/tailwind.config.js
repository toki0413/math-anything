/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: "#0f0f14",
          surface: "#16161e",
          card: "#1c1c26",
          hover: "#242430",
        },
        accent: {
          DEFAULT: "#d4a054",
          dim: "rgba(212,160,84,0.12)",
        },
        accent2: {
          DEFAULT: "#5b8def",
          dim: "rgba(91,141,239,0.12)",
        },
        accent3: {
          DEFAULT: "#6ee7b7",
          dim: "rgba(110,231,183,0.12)",
        },
        text: {
          DEFAULT: "#e4e4ef",
          2: "#9e9eb8",
          3: "#6b6b85",
        },
        border: {
          DEFAULT: "#2a2a3a",
          light: "#3a3a50",
        },
        error: {
          DEFAULT: "#f87171",
          dim: "rgba(248,113,113,0.12)",
        },
        warn: {
          DEFAULT: "#fbbf24",
          dim: "rgba(251,191,36,0.12)",
        },
      },
      fontFamily: {
        display: ["Cormorant Garamond", "Georgia", "serif"],
        body: ["Outfit", "-apple-system", "sans-serif"],
        mono: ["Fira Code", "IBM Plex Mono", "monospace"],
      },
      borderRadius: {
        DEFAULT: "8px",
        lg: "12px",
        xl: "16px",
      },
    },
  },
  plugins: [],
};
