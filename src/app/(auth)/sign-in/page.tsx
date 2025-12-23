'use client'

import { useEffect, useRef } from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import * as z from 'zod'
import { signIn } from 'next-auth/react'
import {
  Form,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useToast } from '@/components/ui/use-toast'
import { signInSchema } from '@/schemas/signInSchema'

export default function SignInForm() {
  const router = useRouter()
  const { toast } = useToast()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const form = useForm<z.infer<typeof signInSchema>>({
    resolver: zodResolver(signInSchema),
    defaultValues: {
      identifier: '',
      password: '',
    },
  })

  const onSubmit = async (data: z.infer<typeof signInSchema>) => {
    const result = await signIn('credentials', {
      redirect: false,
      identifier: data.identifier,
      password: data.password,
    })

    if (result?.error) {
      toast({
        title: 'Login Failed',
        description: 'Invalid credentials',
        variant: 'destructive',
      })
    }

    if (result?.url) {
      router.replace('/dashboard')
    }
  }

  // Canvas animation
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let w = (canvas.width = window.innerWidth)
    let h = (canvas.height = window.innerHeight)

    const colors = ['#8B5CF6', '#EC4899', '#3B82F6', '#22D3EE']
    const orbs: any[] = []

    for (let i = 0; i < 50; i++) {
      orbs.push({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 20 + 10,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        color: colors[Math.floor(Math.random() * colors.length)],
      })
    }

    function animate() {
      if (!ctx) return
      ctx.fillStyle = 'rgba(0,0,0,0.1)'
      ctx.fillRect(0, 0, w, h)

      // Draw orbs
      for (let p of orbs) {
        p.x += p.vx
        p.y += p.vy
        if (p.x < 0 || p.x > w) p.vx *= -1
        if (p.y < 0 || p.y > h) p.vy *= -1

        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r)
        gradient.addColorStop(0, p.color + 'AA')
        gradient.addColorStop(0.7, p.color + '33')
        gradient.addColorStop(1, 'transparent')

        ctx.beginPath()
        ctx.fillStyle = gradient
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fill()
      }

      // Draw lines between close orbs
      for (let i = 0; i < orbs.length; i++) {
        for (let j = i + 1; j < orbs.length; j++) {
          const dx = orbs[i].x - orbs[j].x
          const dy = orbs[i].y - orbs[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 150) {
            ctx.strokeStyle = `rgba(139,92,246,${(1 - dist / 150) * 0.3})`
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(orbs[i].x, orbs[i].y)
            ctx.lineTo(orbs[j].x, orbs[j].y)
            ctx.stroke()
          }
        }
      }

      requestAnimationFrame(animate)
    }

    animate()

    // Resize handling
    const handleResize = () => {
      w = canvas.width = window.innerWidth
      h = canvas.height = window.innerHeight
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden bg-black">
      {/* Canvas for neon orbs */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full"></canvas>

      {/* Form Card */}
      <div className="relative z-10 w-full max-w-md bg-black/70 rounded-2xl p-8 shadow-2xl backdrop-blur-md border border-purple-500">
        <h1 className="text-3xl font-bold text-center mb-2 text-white">
          Welcome to <span className="text-purple-400">Querify</span>
        </h1>
        <p className="text-center text-gray-300 mb-6">
          Sign in to summarize PDFs with AI
        </p>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-5">
            <FormField
              control={form.control}
              name="identifier"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-gray-300">Email / Username</FormLabel>
                  <Input {...field} className="bg-gray-900 text-white border-gray-700" />
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-gray-300">Password</FormLabel>
                  <Input type="password" {...field} className="bg-gray-900 text-white border-gray-700"/>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button className="w-full bg-purple-500 hover:bg-purple-600 text-white">Sign In</Button>
          </form>
        </Form>

        <p className="text-center text-sm text-gray-400 mt-4">
          New to Querify?{' '}
          <Link href="/sign-up" className="text-purple-400 hover:underline">
            Create an account
          </Link>
        </p>
      </div>
    </div>
  )
}
