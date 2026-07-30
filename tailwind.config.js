/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html"],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      colors: {
        brand: {
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a5f',
          950: '#172554',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
  safelist: [
    // Cores dinamicas usadas via JavaScript template literals
    'text-emerald-300', 'text-emerald-400', 'text-emerald-500',
    'text-red-300', 'text-red-400', 'text-red-500',
    'text-amber-400', 'text-amber-500',
    'text-blue-400', 'text-blue-500',
    'text-gray-300', 'text-gray-400', 'text-gray-500', 'text-gray-600',
    'text-purple-400', 'text-purple-500',
    'bg-emerald-500', 'bg-emerald-600',
    'bg-red-500', 'bg-red-600',
    'bg-amber-500', 'bg-amber-600',
    'bg-blue-500', 'bg-blue-600',
    'border-emerald-500', 'border-red-500', 'border-amber-500', 'border-blue-500',
    // Brand color classes
    'text-brand-300', 'text-brand-400', 'border-brand-800/40',
    // Opacity variants usadas em badges dinamicos
    { pattern: /bg-(emerald|red|amber|blue|brand)-900\/(10|20|40)/ },
  ],
}
