'use client';

import React from 'react';
import Link from 'next/link';
import { useSession, signOut } from 'next-auth/react';
import { Button } from './ui/button';
import { User } from 'next-auth';

export default function Navbar() {
  const { data: session } = useSession();
  const user: User | undefined = session?.user;

  return (
    <nav className="fixed top-0 w-full h-16 px-6 bg-black/80 backdrop-blur-md text-white z-30 border-b border-white/10">
      <div className="w-full max-w-7xl mx-auto flex items-center justify-between h-full">
        <Link
          href="/"
          className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent"
        >
          Querify
        </Link>

        {session ? (
          <div className="flex items-center gap-4">
            <span className="truncate max-w-[200px] text-sm text-slate-200">
              Welcome, {user?.username || user?.email}
            </span>
            <Button
              onClick={() => signOut()}
              className="bg-cyan-500 hover:bg-cyan-600 text-white"
            >
              Logout
            </Button>
          </div>
        ) : (
          <Link href="/sign-in">
            <Button className="bg-cyan-500 hover:bg-cyan-600 text-white">
              Login
            </Button>
          </Link>
        )}
      </div>
    </nav>
  );
}
