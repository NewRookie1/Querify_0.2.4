'use client';

import { Button } from '@/components/ui/button';
import Navbar from '@/components/Navbar';

export default function QuerifyHome() {
  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">

      {/* 🌌 BACKGROUND IMAGE (z-0) */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <img
          src="/abstract-hero.png"
          alt=""
          className="
            absolute right-[-160px] top-1/2 -translate-y-1/2
            w-[800px] max-w-none
            opacity-60
            mix-blend-screen
            hero-mask
          "
        />
      </div>

      {/* 🧭 NAVBAR (z-30) */}
      <Navbar />

      {/* HERO CONTENT (z-10) */}
      <section className="relative z-10 flex flex-col md:flex-row items-center justify-between px-10 md:px-20 pt-32 pb-24 gap-16">
        {/* pt-32 = space for fixed navbar */}

        <div className="max-w-xl">
          <h1 className="text-5xl md:text-6xl font-extrabold leading-tight">
            We designed
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-500">
              premium AI
            </span>{' '}
            document tools.
          </h1>

          <p className="text-gray-400 mt-6 text-lg leading-relaxed">
            Upload PDFs, summarize documents, and ask questions instantly using
            powerful AI. Built for students, researchers, and professionals.
          </p>

          <Button
            size="lg"
            className="mt-8 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 hover:opacity-90"
          >
            Get Started
          </Button>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="absolute bottom-6 right-10 flex gap-6 text-gray-400 text-sm z-20">
        <span className="hover:text-white cursor-pointer">Twitter</span>
        <span className="hover:text-white cursor-pointer">LinkedIn</span>
        <span className="hover:text-white cursor-pointer">GitHub</span>
      </footer>

      {/* 🎨 MASK STYLE */}
      <style jsx>{`
        .hero-mask {
          -webkit-mask-image: radial-gradient(
            circle at left center,
            black 0%,
            black 45%,
            transparent 75%
          );
          mask-image: radial-gradient(
            circle at left center,
            black 0%,
            black 45%,
            transparent 75%
          );
        }
      `}</style>
    </div>
  );
}
