
static int
__schro_arith_context_decode_bit (SchroArith * arith, int i)
{
  SchroArithContext *context = arith->contexts + i;

#include "schrooffsets.h"
  __asm__ __volatile__ (
      //calc_count_range(arith);
      "  movzwl a_range(%0), %%ecx\n"
      "  movzwl a_code(%0), %%eax\n"
      "  movzwl a_range+2(%0), %%edx\n"
#if 0
      "  subl $1, %%ecx\n"
      "  subl %%ecx, %%esi\n"
      "  subl %%ecx, %%edx\n"
#else
      "  negl %%ecx\n"
      "  leal 1(%%eax,%%ecx,1), %%eax\n"
      "  leal 1(%%edx,%%ecx,1), %%ecx\n"
#endif
      "  movl %%eax, a_count(%0)\n"
      //"  movl %%ecx, a_range_value(%0)\n"

      //calc_prob0(arith, context);
      "  movzwl c_count(%1), %%eax\n"
      "  addw (c_count + 2)(%1), %%ax\n"
      "  movzwl a_division_factor(%0,%%eax,2), %%eax\n"
#if 1
      "  mulw c_count(%1)\n"
#else
      "  movzwl c_count(%1), %%edx\n"
      "  imul %%edx, %%eax\n"
#endif

      // calc_value()
#if 1
      "  imul %%ecx, %%eax\n"
      "  shrl $16, %%eax\n"
#else
      "  cmp $0x10000, %%ecx\n"
      "  je .Lskipmul\n"
      "  mul %%cx\n"
      "  mov %%dx, %%ax\n"
      ".Lskipmul:\n"
#endif

      "  mov a_count(%0), %%ecx\n"
#if 0
      "  subl %%eax, %%ecx\n"
      "  neg %%ecx\n"
      "  shrl $31, %%ecx\n"
      "  and $1, %%ecx\n"
#else
      "  cmpl %%eax, %%ecx\n"
      "  setg %%cl\n"
      "  movzbl %%cl, %%ecx\n"
#endif

      "  xor $1, %%ecx\n"
      "  addw a_range(%0), %%ax\n"
      "  subw %%cx, %%ax\n"
      "  movw %%ax, a_range(%0,%%ecx,2)\n"
      "  xor $1, %%ecx\n"
      "  addw $1, c_count(%1, %%ecx, 2)\n"
      "  movw %%cx, a_value(%0)\n"

      //maybe_shift_context(context);
#if 0
      "  movw c_count(%1), %%cx\n"
      "  addw c_count + 2(%1), %%cx\n"
      "  shrw $8, %%cx\n"
#if 0
      "  shrw %%cl, c_count(%1)\n"
      "  addw %%cx, c_count(%1)\n"
      "  shrw %%cl, c_count+2(%1)\n"
      "  addw %%cx, c_count+2(%1)\n"
#else
      "  movw %%cx, %%ax\n"
      "  shl $16, %%eax\n"
      "  orl %%ecx, %%eax\n"
      "  movl c_count(%1), %%edx\n"
      "  shrl %%cl, %%edx\n"
      "  addl %%eax, %%edx\n"
      "  and $0x00ff00ff, %%edx\n"
      "  movl %%edx, c_count(%1)\n"
#endif
#else
      "  movw c_count(%1), %%cx\n"
      "  addw c_count + 2(%1), %%cx\n"
      "  cmp $255, %%cx\n"
      "  jle .Lnoshift\n"
#if 0
      "  shrw $1, c_count(%1)\n"
      "  addw $1, c_count(%1)\n"
      "  shrw $1, c_count+2(%1)\n"
      "  addw $1, c_count+2(%1)\n"
#else
      "  movl c_count(%1), %%ecx\n"
      "  shrl $1, %%ecx\n"
      "  addl $0x00010001, %%ecx\n"
      "  and $0x00ff00ff, %%ecx\n"
      "  mov %%ecx, c_count(%1)\n"
#endif
      ".Lnoshift:\n"
#endif

      //fixup_range(arith);
      // i = ((arith->range[1]&0xf000)>>8) | ((arith->range[0]&0xf000)>>12);
      // fixup = arith->fixup_shift[i];
      "  movw a_range + 2(%0), %%ax\n"
      "  shrw $12, %%ax\n"
      "  movw a_range(%0), %%cx\n"
      "  shldw $4, %%cx, %%ax\n"
      "  movzwl a_fixup_shift(%0,%%eax,2), %%eax\n"

      // if (n == 0) return;
      "  test %%eax, %%eax\n"
      "  je .Lfixup_done\n"

      // n = arith->fixup_shift[i] & 0xf;
      "  movl %%eax, %%ecx\n"
      "  andw $0x1f, %%cx\n"
      // arith->range[0] <<= n;
      "  shlw %%cl, a_range(%0)\n"
      // arith->range[1] <<= n;
      // arith->range[1] |= (1<<n)-1;
      "  addw $1, a_range+2(%0)\n"
      "  shlw %%cl, a_range+2(%0)\n"
      "  addw $-1, a_range+2(%0)\n"
      // arith->code <<= n;
      // arith->code |= (arith->nextcode >> ((32-n)&0x1f));
      "  movw a_nextcode+2(%0), %%dx\n"
      "  shldw %%cl, %%dx, a_code(%0)\n"
      // arith->nextcode <<= n;
      "  shll %%cl, a_nextcode(%0)\n"
      // arith->nextbits-=n;
      "  subl %%ecx, a_nextbits(%0)\n"

      // flip = arith->fixup_shift[i] & 0x8000;
      "  andw $0x8000, %%ax\n"
      // arith->code ^= flip;
      "  xorw %%ax, a_code(%0)\n"
      // arith->range[0] ^= flip;
      "  xorw %%ax, a_range(%0)\n"
      // arith->range[1] ^= flip;
      "  xorw %%ax, a_range+2(%0)\n"

      "  cmpw $3, %%cx\n"
      "  jl .Lfixup_nextcode\n"
      ".Lfixup_loop:\n"
      "  movzwl a_range+2(%0), %%eax\n"
      "  shrw $12, %%ax\n"
      "  movw a_range(%0), %%cx\n"
      "  shldw $4, %%cx, %%ax\n"
      "  movzwl a_fixup_shift(%0,%%eax,2), %%eax\n"

      "  test %%eax, %%eax\n"
      "  je .Lfixup_nextcode\n"

      "  movl %%eax, %%ecx\n"
      "  andw $0x1f, %%cx\n"
      "  shlw %%cl, a_range(%0)\n"
      "  addw $1, a_range+2(%0)\n"
      "  shlw %%cl, a_range+2(%0)\n"
      "  addw $-1, a_range+2(%0)\n"
      "  movw a_nextcode+2(%0), %%dx\n"
      "  shldw %%cl, %%dx, a_code(%0)\n"
      "  shll %%cl, a_nextcode(%0)\n"
      "  subl %%ecx, a_nextbits(%0)\n"

      "  andw $0x8000, %%ax\n"
      "  xorw %%ax, a_code(%0)\n"
      "  xorw %%ax, a_range(%0)\n"
      "  xorw %%ax, a_range+2(%0)\n"

      "  cmpw $3, %%cx\n"
      "  jge .Lfixup_loop\n"
      ".Lfixup_nextcode:\n"
      "  movl $24, %%ecx\n"
      "  subl a_nextbits(%0), %%ecx\n"
      "  jb .Lfixup_done\n"

      "  movl a_dataptr(%0), %%eax\n"
      "  cmpl a_maxdataptr(%0), %%eax\n"
      "  jge .Lpast_end\n"

      "  movzbl 0(%%eax), %%edx\n"
      "  jmp .Lcont\n"

      ".Lpast_end:\n"
      "  movl $0xff, %%edx\n"

      ".Lcont:\n"
      "  shll %%cl, %%edx\n"
      "  orl %%edx, a_nextcode(%0)\n"

      "  addl $8, a_nextbits(%0)\n"
      "  addl $1, a_dataptr(%0)\n"
      "  addl $1, a_offset(%0)\n"
      "  jmp .Lfixup_nextcode\n"

      ".Lfixup_done:\n"

      :
      : "r" (arith), "r" (context)
      : "eax", "ecx", "edx", "memory");

  return arith->value;
}

