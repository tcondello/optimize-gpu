#!/usr/bin/env python3
"""
Builds a deterministic, programmatically verified 150-prompt reasoning benchmark
(`data/val_prompts.jsonl`) across 5 categories (30 prompts each) designed to test:
1. Macro reasoning accuracy (exact ground-truth answer computed by Python math/simulators),
2. Reasoning token bloat & circular self-doubt loops ("Wait,", "Hmm,", n-gram loops),
3. Premature `</think>` early-exit failures on deceptive / multi-constraint problems.
"""

import json
import math
from pathlib import Path

OUT_PATH = Path("/Users/tim/Code/optimize-gpu/data/val_prompts.jsonl")
SUFFIX = "\n\nProvide your final result after thinking on the last line in the exact format: Final Answer: <number>"


def gen_gsm_math():
    items = []
    # 10 multi-rate production & defect word problems
    configs_a = [
        ("Elena", "widgets", 20, 6, 4, 25, 3),
        ("Marcus", "circuit boards", 40, 5, 3, 15, 4),
        ("Priya", "ceramic mugs", 15, 8, 5, 20, 2),
        ("Liam", "solar panels", 12, 7, 4, 25, 3),
        ("Sofia", "watch gears", 30, 6, 5, 20, 5),
        ("Noah", "leather belts", 20, 9, 6, 30, 4),
        ("Maya", "glassware sets", 16, 5, 4, 25, 3),
        ("Owen", "wooden chairs", 25, 6, 3, 20, 2),
        ("Chloe", "bicycles", 10, 8, 5, 40, 3),
        ("Ethan", "drone frames", 50, 5, 4, 18, 3),
    ]
    for name, item, rate1, hrs1, hrs2, bonus_pct, defect_every in configs_a:
        assert (rate1 * (100 + bonus_pct)) % 100 == 0
        rate2 = (rate1 * (100 + bonus_pct)) // 100
        gross = rate1 * hrs1 + rate2 * hrs2
        defects = gross // defect_every
        good = gross - defects
        p = (
            f"{name} manufactures {item.strip()}. For the first {hrs1} hours of a shift, {name} produces "
            f"{rate1} {item.strip()} per hour. After a machine tune-up, hourly output increases by {bonus_pct}% "
            f"for the remaining {hrs2} hours. Quality control then rejects 1 out of every {defect_every} "
            f"produced items (rounded down to the nearest whole item). How many {item.strip()} pass inspection?"
            + SUFFIX
        )
        items.append(("gsm_math", p, str(good)))

    # 10 tiered store / discount / tax / shipping integer problems
    configs_b = [
        (4, 35, 3, 20, 10, 15, 12),
        (5, 28, 4, 15, 20, 10, 18),
        (6, 40, 2, 30, 15, 20, 10),
        (3, 60, 5, 24, 10, 25, 15),
        (8, 25, 4, 25, 20, 15, 14),
        (5, 48, 3, 20, 10, 30, 9),
        (7, 30, 3, 30, 15, 18, 11),
        (4, 55, 6, 15, 20, 22, 16),
        (9, 20, 4, 30, 10, 12, 8),
        (6, 45, 5, 18, 20, 24, 15),
    ]
    for q1, p1, q2, p2, disc_pct, coupon, ship in configs_b:
        sub = q1 * p1 + q2 * p2
        after_disc = (sub * (100 - disc_pct)) // 100
        assert (sub * (100 - disc_pct)) % 100 == 0
        total = after_disc - coupon + ship
        p = (
            f"A lab orders {q1} sensors at ${p1} each and {q2} cables at ${p2} each. "
            f"A bulk discount takes {disc_pct}% off the combined subtotal first, then a flat ${coupon} "
            f"rebate voucher is subtracted, and finally a flat ${ship} hazmat shipping fee is added. "
            f"What is the final bill in dollars?"
            + SUFFIX
        )
        items.append(("gsm_math", p, str(total)))

    # 10 train / tank / opposing-rate meeting & fuel problems
    configs_c = [
        (420, 65, 75, 40, 3),
        (540, 80, 100, 60, 4),
        (360, 50, 70, 60, 2),
        (600, 70, 80, 75, 5),
        (480, 55, 65, 60, 3),
        (720, 85, 95, 90, 4),
        (300, 45, 55, 50, 2),
        (660, 105, 115, 110, 6),
        (450, 70, 80, 75, 3),
        (840, 95, 115, 105, 5),
    ]
    for dist, v1, v2, drone_v, per_km in configs_c:
        # Train A gets 1 hour head start? Let's make exact integer meeting time:
        # Both start simultaneously from `dist` apart: t = dist / (v1 + v2)
        assert dist % (v1 + v2) == 0
        t_hrs = dist // (v1 + v2)
        ans = t_hrs * drone_v + per_km * t_hrs
        p = (
            f"Two freight trains start simultaneously from stations {dist} km apart and travel toward each other "
            f"on parallel tracks at constant speeds of {v1} km/h and {v2} km/h. A scout drone launches when they "
            f"depart and flies at {drone_v} km/h until the trains pass each other, plus logs {per_km} telemetry "
            f"packets at the end of each hour of flight. What is the sum of the total kilometers flown by the "
            f"drone and the total telemetry packets logged?"
            + SUFFIX
        )
        items.append(("gsm_math", p, str(ans)))

    assert len(items) == 30
    return items


def gen_algebra_numtheory():
    items = []
    # 10 Chinese Remainder / simultaneous congruence problems
    crt_specs = [
        (3, 2, 5, 3, 7, 2, 100, 220),
        (4, 1, 5, 4, 7, 3, 150, 300),
        (3, 1, 7, 5, 11, 4, 200, 450),
        (5, 2, 6, 5, 7, 1, 250, 470),
        (4, 3, 7, 2, 9, 5, 300, 560),
        (3, 2, 8, 5, 11, 7, 300, 580),
        (5, 4, 7, 6, 8, 3, 200, 500),
        (6, 1, 7, 4, 11, 9, 400, 900),
        (4, 1, 9, 7, 11, 3, 350, 750),
        (5, 3, 8, 1, 9, 4, 300, 700),
    ]
    for m1, r1, m2, r2, m3, r3, lo, hi in crt_specs:
        sols = [
            x
            for x in range(lo, hi + 1)
            if x % m1 == r1 and x % m2 == r2 and x % m3 == r3
        ]
        assert len(sols) >= 1, f"No sol for {(m1,r1,m2,r2,m3,r3,lo,hi)}"
        ans = sols[0]
        p = (
            f"Find the smallest integer N >= {lo} such that N leaves a remainder of {r1} when divided by {m1}, "
            f"a remainder of {r2} when divided by {m2}, and a remainder of {r3} when divided by {m3}."
            + SUFFIX
        )
        items.append(("algebra_numtheory", p, str(ans)))

    # 10 modular exponentiation + sum problems
    mod_specs = [
        (7, 2026, 13, 5, 99),
        (3, 2025, 17, 11, 50),
        (5, 1001, 19, 7, 40),
        (2, 2030, 23, 9, 60),
        (6, 555, 29, 4, 77),
        (11, 888, 31, 3, 64),
        (9, 777, 37, 8, 45),
        (4, 1234, 41, 6, 52),
        (8, 999, 43, 12, 33),
        (10, 1500, 47, 15, 28),
    ]
    for base, exp, modp, base2, exp2 in mod_specs:
        val = (pow(base, exp, modp) + pow(base2, exp2, modp)) % modp
        p = (
            f"Compute the remainder when ({base}^{exp} + {base2}^{exp2}) is divided by {modp}."
            + SUFFIX
        )
        items.append(("algebra_numtheory", p, str(val)))

    # 10 combinatorics / inclusion-exclusion / divisor count problems
    ie_specs = [
        (600, 4, 6, 15),
        (500, 3, 5, 12),
        (840, 6, 10, 14),
        (720, 4, 9, 10),
        (900, 5, 6, 9),
        (1000, 4, 6, 10),
        (450, 3, 6, 15),
        (750, 5, 10, 12),
        (660, 4, 6, 11),
        (960, 6, 8, 15),
    ]
    for limit, a, b, c in ie_specs:
        cnt = sum(
            1
            for k in range(1, limit + 1)
            if (k % a == 0 or k % b == 0) and (k % c != 0)
        )
        p = (
            f"How many positive integers from 1 to {limit} (inclusive) are divisible by {a} or by {b} "
            f"(or both), yet are NOT divisible by {c}?"
            + SUFFIX
        )
        items.append(("algebra_numtheory", p, str(cnt)))

    assert len(items) == 30
    return items


def gen_logic_knights_grid():
    items = []
    # 15 4-person Knights (always tell truth) and Knaves (always lie) unique-assignment puzzles
    # Encode person bitmask 0..15; find statements with a UNIQUE satisfying bitmask and ask `8*A + 4*B + 2*C + D`!
    names = ["Ava", "Ben", "Cole", "Dana"]

    def make_kk_puzzles_15():
        # Enumerate clean, natural 4-statement Knights & Knaves puzzles and keep only those with a UNIQUE solution!
        found = []
        seen_sigs = set()
        for k_ava in range(5):
            for s1_kind in range(4):
                for s2_kind in range(4):
                    for s3_kind in range(4):
                        s0_txt = f"Exactly {k_ava} of the four of us are knights."
                        s1_txt = [
                            "Cole is a knight.",
                            "Cole is a knave.",
                            "Ava and Cole are both knights.",
                            "At least one of Ava or Dana is a knave.",
                        ][s1_kind]
                        s2_txt = [
                            "Dana is a knight.",
                            "Dana is a knave.",
                            "Ben and Dana are the same type.",
                            "Exactly one of Ava or Ben is a knight.",
                        ][s2_kind]
                        s3_txt = [
                            "Ava is a knight.",
                            "Ben is a knave.",
                            "Both Ben and Cole are knaves.",
                            "Ava and Cole are of different types.",
                        ][s3_kind]

                        valid = []
                        for m in range(16):
                            b = [(m >> (3 - i)) & 1 for i in range(4)]
                            t0 = (sum(b) == k_ava)
                            t1 = [
                                b[2] == 1,
                                b[2] == 0,
                                (b[0] == 1 and b[2] == 1),
                                (b[0] == 0 or b[3] == 0),
                            ][s1_kind]
                            t2 = [
                                b[3] == 1,
                                b[3] == 0,
                                b[1] == b[3],
                                (b[0] + b[1]) == 1,
                            ][s2_kind]
                            t3 = [
                                b[0] == 1,
                                b[1] == 0,
                                (b[1] == 0 and b[2] == 0),
                                b[0] != b[2],
                            ][s3_kind]
                            if b == [int(t0), int(t1), int(t2), int(t3)]:
                                valid.append(m)
                        if len(valid) == 1:
                            sol = valid[0]
                            sig = (sol, k_ava, s1_kind)
                            if sig not in seen_sigs:
                                seen_sigs.add(sig)
                                found.append((sol, s0_txt, s1_txt, s2_txt, s3_txt))
        # Pick 15 diverse unique-solution puzzles across distinct bitmasks
        chosen = []
        used_sols = set()
        for rec in found:
            if rec[0] not in used_sols:
                used_sols.add(rec[0])
                chosen.append(rec)
                if len(chosen) == 15:
                    return chosen
        for rec in found:
            if rec not in chosen:
                chosen.append(rec)
                if len(chosen) == 15:
                    return chosen
        return chosen

    kk_puzzles = make_kk_puzzles_15()
    assert len(kk_puzzles) == 15, f"Only got {len(kk_puzzles)} unique KK puzzles"
    for code_val, s0_txt, s1_txt, s2_txt, s3_txt in kk_puzzles:
        p = (
            f"On an island, every inhabitant is either a Knight (who always tells the truth) or a Knave "
            f"(who always lies). Four inhabitants—Ava, Ben, Cole, and Dana—make these statements:\n"
            f"- Ava says: \"{s0_txt}\"\n"
            f"- Ben says: \"{s1_txt}\"\n"
            f"- Cole says: \"{s2_txt}\"\n"
            f"- Dana says: \"{s3_txt}\"\n"
            f"Let A, B, C, D each be 1 if Ava, Ben, Cole, Dana respectively is a Knight, and 0 if a Knave. "
            f"Compute the integer 8*A + 4*B + 2*C + D."
            + SUFFIX
        )
        items.append(("logic_knights_grid", p, str(code_val)))

    # 15 Linear Seating / Ranking Constraint Deduction puzzles (5 items ranked 1..5, compute weighted checksum)
    import itertools

    perms = list(itertools.permutations([1, 2, 3, 4, 5]))
    chosen_perms = perms[::8][:15]
    assert len(chosen_perms) == 15
    for idx, target in enumerate(chosen_perms):
        # target = (rP, rQ, rR, rS, rT) distinct positions 1..5 from left (1) to right (5)
        rP, rQ, rR, rS, rT = target
        d_pq = rP - rQ
        d_rs = abs(rR - rS)
        # Build clues that uniquely identify `target` among all 120 permutations!
        clue_list = [
            f"1. Box P is positioned {abs(d_pq)} slot(s) to the {'right' if d_pq > 0 else 'left'} of Box Q (so P - Q = {d_pq}).",
            f"2. The distance |R - S| between Box R and Box S is {d_rs}, with Box R {'to the left of' if rR < rS else 'to the right of'} Box S.",
            f"3. Box R is at slot {rR} or slot {((rR % 5) + 1)}, and Box R is {'to the left of' if rR < rT else 'to the right of'} Box T.",
            f"4. Box S is {'strictly left of' if rS < rQ else 'strictly right of'} Box Q, and Box T is at slot {rT}.",
        ]
        # Verify uniqueness across all 120 permutations!
        matches = []
        for pm in perms:
            p_, q_, r_, s_, t_ = pm
            c1 = (p_ - q_ == d_pq)
            c2 = (abs(r_ - s_) == d_rs) and ((r_ < s_) == (rR < rS))
            c3 = (r_ in (rR, (rR % 5) + 1)) and ((r_ < t_) == (rR < rT))
            c4 = ((s_ < q_) == (rS < rQ)) and (t_ == rT)
            if c1 and c2 and c3 and c4:
                matches.append(pm)
        assert matches == [target], f"Matches={matches} for {target}"
        checksum = 100 * rP + 10 * rQ + rR
        p = (
            f"Five labeled boxes (P, Q, R, S, T) occupy slots 1, 2, 3, 4, 5 from left to right (one box per slot):\n"
            + "\n".join(clue_list)
            + f"\nDetermine every box's slot number and compute 100*(slot of P) + 10*(slot of Q) + (slot of R)."
            + SUFFIX
        )
        items.append(("logic_knights_grid", p, str(checksum)))

    assert len(items) == 30
    return items


def gen_crt_deceptive():
    items = []
    # 10 Snail / Well climb-and-slip boundary problems (classic premature-`</think>` off-by-1 trap!)
    snail_specs = [
        (30, 7, 4),
        (45, 9, 5),
        (52, 10, 6),
        (38, 8, 3),
        (61, 11, 6),
        (29, 6, 2),
        (75, 12, 7),
        (44, 9, 4),
        (67, 13, 8),
        (50, 8, 5),
    ]
    for H, up, down in snail_specs:
        pos = 0
        day = 0
        while True:
            day += 1
            pos += up
            if pos >= H:
                break
            pos -= down
        p = (
            f"An inspection robot climbs out of a vertical shaft that is {H} meters deep, starting at 0 meters "
            f"at dawn on Day 1. Each daytime it climbs up {up} meters; if it has reached or passed {H} meters, "
            f"it exits immediately, otherwise each night it slides back down {down} meters. On what day number "
            f"does the robot reach the top of the shaft?"
            + SUFFIX
        )
        items.append(("crt_deceptive", p, str(day)))

    # 10 Worst-case Pigeonhole draw problems (another classic trap!)
    ph_specs = [
        (12, 15, 9, 4),
        (10, 14, 18, 5),
        (8, 20, 12, 6),
        (15, 15, 15, 7),
        (6, 11, 19, 5),
        (9, 13, 17, 8),
        (5, 25, 20, 6),
        (14, 10, 16, 9),
        (7, 18, 22, 8),
        (11, 12, 13, 10),
    ]
    for r, g, b, k in ph_specs:
        # Worst case without `k` of the SAME color: take min(c, k-1) from each color, + 1
        worst = min(r, k - 1) + min(g, k - 1) + min(b, k - 1) + 1
        p = (
            f"A bin in a dark room contains {r} red resistors, {g} green resistors, and {b} blue resistors. "
            f"What is the minimum number of resistors you must draw blindly to be 100% certain that you have "
            f"at least {k} resistors of the SAME color?"
            + SUFFIX
        )
        items.append(("crt_deceptive", p, str(worst)))

    # 10 Harmonic-mean round-trip speed / bacteria doubling / fencepost interval problems
    misc_specs = [
        (30, 60, "speed"),
        (40, 60, "speed"),
        (20, 80, "speed"),
        (60, 120, "speed"),
        (24, 72, "speed"),
        (48, 4, "halving"),
        (60, 8, "halving"),
        (36, 16, "halving"),
        (120, 6, "fencepost"),
        (210, 14, "fencepost"),
    ]
    for a, b, kind in misc_specs:
        if kind == "speed":
            assert (2 * a * b) % (a + b) == 0
            ans = (2 * a * b) // (a + b)
            p = (
                f"A courier drives from Town X to Town Y at a constant speed of {a} km/h and immediately drives "
                f"back along the exact same route from Town Y to Town X at {b} km/h. What is the courier's "
                f"average speed (in km/h) for the entire round trip?"
                + SUFFIX
            )
        elif kind == "halving":
            # Doubles every minute, full at `a` minutes; when is it `1/b` full? `b` is power of 2!
            steps = int(round(math.log2(b)))
            assert (1 << steps) == b
            ans = a - steps
            p = (
                f"A culture of algae in a bioreactor doubles in volume every single minute. If the tank starts "
                f"partially seeded at t = 0 and becomes completely full at t = {a} minutes, at what minute t "
                f"was the tank exactly 1/{b} full?"
                + SUFFIX
            )
        else:
            assert a % b == 0
            # Straight path of length `a` meters with posts every `b` meters at BOTH ends AND BOTH sides of path!
            ans = 2 * ((a // b) + 1)
            p = (
                f"A straight walkway is {a} meters long. Lamp posts are placed along BOTH sides of the walkway "
                f"every {b} meters, including at both the starting end (0 m) and the finishing end ({a} m) on "
                f"each side. How many lamp posts are needed in total?"
                + SUFFIX
            )
        items.append(("crt_deceptive", p, str(ans)))

    assert len(items) == 30
    return items


def gen_algo_trace():
    items = []
    # 10 Collatz / stateful integer loop trace problems
    seeds_a = [7, 11, 13, 15, 17, 19, 21, 23, 25, 26]
    for s0 in seeds_a:
        x = s0
        acc = 0
        for step in range(1, 9):
            if x % 2 == 0:
                x = x // 2 + step
            else:
                x = 3 * x - 1
            acc += x
        p = (
            f"Trace the following loop starting with x = {s0} and acc = 0:\n"
            f"For step = 1, 2, ..., 8:\n"
            f"  - If x is even, set x = (x / 2) + step\n"
            f"  - Otherwise (x is odd), set x = 3*x - 1\n"
            f"  - Then set acc = acc + x\n"
            f"What is the final value of acc after step = 8 finishes?"
            + SUFFIX
        )
        items.append(("algo_trace", p, str(acc)))

    # 10 Bitwise XOR / shift / popcount trace problems
    bit_specs = [
        (29, 14, 5),
        (45, 27, 6),
        (53, 19, 4),
        (37, 22, 5),
        (61, 35, 6),
        (42, 25, 5),
        (50, 31, 4),
        (33, 18, 6),
        (58, 29, 5),
        (47, 21, 4),
    ]
    for a0, b0, rounds in bit_specs:
        a, b = a0, b0
        for r in range(1, rounds + 1):
            a = (a ^ (b + r)) & 63
            b = ((b << 1) ^ a) & 63
        ans = a + b
        p = (
            f"Let a = {a0} and b = {b0}. Execute {rounds} iterations for r = 1, 2, ..., {rounds} "
            f"(where `^` is bitwise XOR, `<< 1` multiplies by 2, and `& 63` keeps the lowest 6 bits / mod 64):\n"
            f"  a = (a ^ (b + r)) & 63\n"
            f"  b = ((b << 1) ^ a) & 63   (using the newly updated `a`)\n"
            f"What is the value of a + b after all {rounds} iterations?"
            + SUFFIX
        )
        items.append(("algo_trace", p, str(ans)))

    # 10 Recurrence / Dynamic Programming sequence trace problems
    rec_specs = [
        (1, 2, 2, -1, 7),
        (2, 1, 3, -2, 7),
        (1, 3, 2, 1, 7),
        (3, 2, 2, -1, 8),
        (2, 3, 1, 2, 8),
        (1, 4, 2, -1, 7),
        (4, 1, 2, 1, 7),
        (2, 5, 1, 1, 9),
        (3, 1, 3, -1, 6),
        (1, 2, 3, -1, 6),
    ]
    for f0, f1, c1, c2, n_target in rec_specs:
        seq = [f0, f1]
        for k in range(2, n_target + 1):
            seq.append(c1 * seq[-1] + c2 * seq[-2] + k)
        ans = seq[n_target]
        op_str = f"+ {c2}*F(k-2)" if c2 >= 0 else f"- {abs(c2)}*F(k-2)"
        p = (
            f"A sequence is defined by F(0) = {f0}, F(1) = {f1}, and for every k >= 2:\n"
            f"  F(k) = {c1}*F(k-1) {op_str} + k\n"
            f"Compute the exact value of F({n_target})."
            + SUFFIX
        )
        items.append(("algo_trace", p, str(ans)))

    assert len(items) == 30
    return items


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_items = (
        gen_gsm_math()[:20]
        + gen_algebra_numtheory()[:20]
        + gen_logic_knights_grid()[:20]
        + gen_crt_deceptive()[:20]
        + gen_algo_trace()[:20]
    )
    assert len(all_items) == 100
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for idx, (cat, prompt, ans) in enumerate(all_items):
            rec = {
                "id": idx,
                "category": cat,
                "prompt": prompt,
                "answer": ans,
                "answer_type": "integer",
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(all_items)} deterministic verified reasoning prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()
