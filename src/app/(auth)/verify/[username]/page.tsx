'use client';

import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { ApiResponse } from '@/types/ApiResponse';
import axios, { AxiosError } from 'axios';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useMemo, useRef, useState } from 'react';

type Dot = {
  top: string;
  left: string;
  delay: string;
  duration: string;
  dx: number;
  dy: number;
};

export default function VerifyAccount() {
  const router = useRouter();
  const params = useParams<{ username: string }>();
  const { toast } = useToast();

  /* ---------- OTP STATE ---------- */
  const [otp, setOtp] = useState<string[]>(Array(6).fill(''));
  const [submitting, setSubmitting] = useState(false);
  const inputsRef = useRef<(HTMLInputElement | null)[]>([]);

  useEffect(() => {
    inputsRef.current[0]?.focus();
  }, []);

  /* ---------- AUTO SUBMIT WHEN OTP COMPLETE ---------- */
  useEffect(() => {
    if (otp.every((d) => d !== '') && !submitting) {
      submit();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [otp]);

  /* ---------- RANDOM FLOATING DOTS ---------- */
  const dots = useMemo<Dot[]>(() => {
    return Array.from({ length: 28 }).map(() => {
      const angle = Math.random() * Math.PI * 2;
      const distance = 80 + Math.random() * 120;

      return {
        top: `${Math.random() * 95}%`,
        left: `${Math.random() * 95}%`,
        delay: `${Math.random() * 4}s`,
        duration: `${6 + Math.random() * 6}s`,
        dx: Math.cos(angle) * distance,
        dy: Math.sin(angle) * distance,
      };
    });
  }, []);

  /* ---------- OTP HANDLERS ---------- */
  const handleChange = (v: string, i: number) => {
    if (!/^\d?$/.test(v)) return;

    const next = [...otp];
    next[i] = v;
    setOtp(next);

    if (v && i < 5) {
      inputsRef.current[i + 1]?.focus();
    }
  };

  const handleBackspace = (
    e: React.KeyboardEvent<HTMLInputElement>,
    i: number
  ) => {
    if (e.key === 'Backspace') {
      const next = [...otp];
      if (otp[i]) {
        next[i] = '';
        setOtp(next);
      } else if (i > 0) {
        inputsRef.current[i - 1]?.focus();
      }
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
    const value = e.clipboardData.getData('text').slice(0, 6);
    if (!/^\d+$/.test(value)) return;

    setOtp(value.split(''));
    inputsRef.current[5]?.focus();
    e.preventDefault();
  };

  /* ---------- SUBMIT ---------- */
  const submit = async () => {
    const code = otp.join('');
    if (code.length !== 6 || submitting) return;

    setSubmitting(true);

    try {
      const res = await axios.post<ApiResponse>('/api/verify-code', {
        username: params.username,
        code,
      });

      toast({ title: 'Success', description: res.data.message });
      router.replace('/sign-in');
    } catch (err) {
      const e = err as AxiosError<ApiResponse>;
      toast({
        title: 'Verification failed',
        description: e.response?.data.message ?? 'Try again',
        variant: 'destructive',
      });
      setSubmitting(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-black">

      {/* 🌌 BACKGROUND */}
      <div className="absolute inset-0 gradient-bg" />

      {/* ✨ FLOATING DOTS */}
      <div className="absolute inset-0 pointer-events-none">
        {dots.map((dot, i) => (
          <span
            key={i}
            className="dot"
            style={{
              top: dot.top,
              left: dot.left,
              animationDelay: dot.delay,
              animationDuration: dot.duration,
              ['--dx' as any]: `${dot.dx}px`,
              ['--dy' as any]: `${dot.dy}px`,
            }}
          />
        ))}
      </div>

      {/* 🧾 CARD */}
      <div className="relative z-10 flex min-h-screen items-center justify-center">
        <div className="w-full max-w-md rounded-2xl bg-white/85 backdrop-blur-xl p-8 shadow-2xl">
          <h1 className="text-3xl font-bold text-center mb-2">
            Verify your email
          </h1>
          <p className="text-center text-sm text-gray-600 mb-6">
            Enter the 6-digit code sent to your email
          </p>

          {/* ✅ OTP INPUTS */}
          <div className="flex justify-center gap-3 mb-6">
            {otp.map((_, i) => (
              <input
                key={i}
                ref={(el) => (inputsRef.current[i] = el)}
                value={otp[i]}
                maxLength={1}
                inputMode="numeric"
                autoComplete="one-time-code"
                className="w-12 h-12 rounded-lg border text-center text-lg font-semibold focus:outline-none focus:ring-2 focus:ring-blue-500"
                onChange={(e) => handleChange(e.target.value, i)}
                onKeyDown={(e) => handleBackspace(e, i)}
                onPaste={handlePaste}
                disabled={submitting}
              />
            ))}
          </div>

          <Button
            onClick={submit}
            disabled={submitting}
            className="w-full bg-gradient-to-r from-blue-700 to-black"
          >
            {submitting ? 'Verifying…' : 'Continue'}
          </Button>
        </div>
      </div>

      {/* 🎨 STYLES */}
      <style jsx>{`
        .gradient-bg {
          background: linear-gradient(180deg, black, darkblue, cyan);
          background-size: 300% 300%;
          animation: gradientMove 4s ease infinite;
        }

        @keyframes gradientMove {
          0% { background-position: 24% 0%; }
          50% { background-position: 0% 15%; }
          100% { background-position: 10% 0%; }
        }

        .dot {
          position: absolute;
          width: 3px;
          height: 3px;
          background: rgba(191,219,254,0.9);
          border-radius: 50%;
          animation: float linear infinite;
          box-shadow: 0 0 6px rgba(191,219,254,0.8);
        }

        @keyframes float {
          0% {
            transform: translate(0, 0);
            opacity: 0;
          }
          15% { opacity: 1; }
          100% {
            transform: translate(var(--dx), var(--dy));
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
}
