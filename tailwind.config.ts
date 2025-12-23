import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },

      /* 🔥 Custom animation for rotating squares */
      keyframes: {
        spinFastSlow: {
          '0%': { transform: 'rotate(0deg)' },
          '35%': { transform: 'rotate(360deg)' }, // fast spin
          '100%': { transform: 'rotate(420deg)' }, // slow finish
        },
      },
      animation: {
        'spin-fast-slow':
          'spinFastSlow 1.6s cubic-bezier(0.4, 0, 0.2, 1)',
      },
    },
  },
  plugins: [],
};

export default config;
