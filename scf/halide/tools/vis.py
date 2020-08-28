#!/usr/bin/env python3

from manim import *

class Decompose(Scene):
    # storyboard:
    # original (tensor-expression) code on the upper left
    # original loops on the lower left
    # original iteration space (square) on the lower right
    # original complexity (N^2) in the upper right

    # iteration space square splits into 3: lower triangle, diagonal, upper triangle
    # loops split into 3: i < j, i == j, i > j

    # lower triangle merges with upper triangle
    # i < j merges with i > j
    # complexity divides by 2

    def construct(self):

        # loop structure in top left
        loopsA = Text("loops:", font="Times")
        loopsS1 = TextMobject('for i $\\in$ (0..N):')
        loopsS2 = TextMobject('for j $\\in$ (0..N):')
        loopsS3 = TextMobject('$f(i, j) = g(i, j)$')
        loopsA.to_corner(UP+LEFT)
        loopsS1.next_to(loopsA , DOWN).align_to(loopsA , LEFT).scale(0.8)
        loopsS2.next_to(loopsS1, DOWN).align_to(loopsS1, LEFT).scale(0.8).shift(0.2*RIGHT)
        loopsS3.next_to(loopsS2, DOWN).align_to(loopsS2, LEFT).scale(0.8).shift(0.2*RIGHT)
        self.add(loopsA, loopsS1, loopsS2, loopsS3)

        # iteration space square in bottom right
        spaceA = Text("iteration space:", font="Times")
        space1 = Square()
        space2 = Square()
        space3 = Square()
        space1.set_fill(PINK, opacity=0.5)
        space2.set_fill(GOLD, opacity=0.5)
        space3.set_fill(GREEN, opacity=0.5)
        spaceA.to_corner(UP+RIGHT)
        space1.next_to(spaceA, DOWN)
        space2.align_to(space1, UP+LEFT)
        space3.align_to(space1, UP+LEFT)
        self.add(spaceA, space1, space2, space3)

        # draw arrows
        arrow1S = Arrow(loopsS3, space2.get_edge_center(LEFT))
        self.add(arrow1S)

        # intro period
        self.wait(5)

        # split up iteration spaces
        spaceA2 = Text("iteration space:", font="Times")
        spaceU = Polygon(np.array([0,1,0]), np.array([1,0,0]), np.array([1,1,0]))
        spaceD = Line(np.array([1,0,0]), np.array([0,1,0]))
        spaceL = Polygon(np.array([0,0,0]), np.array([0,1,0]), np.array([1,0,0]))
        spaceU.scale(2).set_fill(PINK, opacity=0.5)
        spaceD.scale(2).set_fill(GOLD, opacity=0.5)
        spaceL.scale(2).set_fill(GREEN, opacity=0.5)
        spaceA2.to_corner(UP+RIGHT)
        spaceU.next_to(spaceA2, DOWN)
        spaceD.next_to(spaceU, DOWN)
        spaceL.next_to(spaceD, DOWN)

        # split up loops
        loopsA2 = Text("loops:", font="Times")
        loopsU1 = TextMobject('for i $\\in$ (0..N):')
        loopsU2 = TextMobject('for j $\\in$ (0..i-1):')
        loopsU3 = TextMobject('$f(i, j) = g(i, j)$')
        loopsU4 = TextMobject('$f(j, i) = g(i, j)$')
        loopsUDline = Line(np.array([0,0,0]), np.array([1,0,0]))
        loopsD1 = TextMobject('for i $\\in$ (0..N):')
        loopsD2 = TextMobject('$f(i, i) = g(i, i)$')
        loopsDLline = Line(np.array([0,0,0]), np.array([1,0,0]))
        loopsL1 = TextMobject('for i $\\in$ (0..N):')
        loopsL2 = TextMobject('for j $\\in$ (i+1..N):')
        loopsL3 = TextMobject('$f(i, j) = g(i, j)$')
        loopsA2.to_corner(UP+LEFT)
        loopsU1    .next_to(loopsA2    , DOWN).align_to(loopsA2, LEFT).scale(0.8)
        loopsU2    .next_to(loopsU1    , DOWN).align_to(loopsU1, LEFT).scale(0.8).shift(0.2*RIGHT)
        loopsU3    .next_to(loopsU2    , DOWN).align_to(loopsU2, LEFT).scale(0.8).shift(0.2*RIGHT)
        loopsU4    .next_to(loopsU3    , DOWN).align_to(loopsU2, LEFT).scale(0.8).shift(0.2*RIGHT)
        loopsUDline.next_to(loopsU4    , DOWN).align_to(loopsU1, LEFT).scale(0.8)
        loopsD1    .next_to(loopsUDline, DOWN).align_to(loopsU1, LEFT).scale(0.8)
        loopsD2    .next_to(loopsD1    , DOWN).align_to(loopsD1, LEFT).scale(0.8).shift(0.2*RIGHT)
        loopsDLline.next_to(loopsD2    , DOWN).align_to(loopsU1, LEFT).scale(0.8)
        loopsL1    .next_to(loopsDLline, DOWN).align_to(loopsU1, LEFT).scale(0.8)
        loopsL2    .next_to(loopsL1    , DOWN).align_to(loopsL1, LEFT).scale(0.8).shift(0.2*RIGHT)
        loopsL3    .next_to(loopsL2    , DOWN).align_to(loopsL2, LEFT).scale(0.8).shift(0.2*RIGHT)

        # draw arrows from loops to spaces
        arrowL = Arrow(loopsL3, spaceL.get_edge_center(LEFT))
        arrowD = Arrow(loopsD2, spaceD.get_edge_center(LEFT))
        arrowU = Arrow(loopsU3, spaceU.get_edge_center(LEFT))

        self.play(ReplacementTransform(space1, spaceU) , ReplacementTransform(space2, spaceD) , ReplacementTransform(space3, spaceL), ReplacementTransform(spaceA, spaceA2),
                  ReplacementTransform(loopsS1, loopsL1), ReplacementTransform(loopsS2, loopsL2), ReplacementTransform(loopsS3, loopsL3), ReplacementTransform(loopsA, loopsA2), ReplacementTransform(arrow1S, arrowL),
                  FadeIn(loopsUDline), FadeIn(loopsD1), FadeIn(loopsD2), FadeIn(loopsDLline), FadeIn(loopsU1), FadeIn(loopsU2), FadeIn(loopsU3), FadeIn(arrowD), FadeIn(arrowU))
        self.wait(3)

        # transpose L
        loopsLTA = Text("(transpose)", font="Times")
        loopsLT2 = TextMobject('for j $\\in$ (0..i-1):')
        loopsLT3 = TextMobject('$f(j, i) = g(i, j)$')
        loopsLT2.next_to(loopsL1 , DOWN).align_to(loopsL1 , LEFT).scale(0.8).shift(0.2*RIGHT)
        loopsLT3.next_to(loopsLT2, DOWN).align_to(loopsLT2, LEFT).scale(0.8).shift(0.2*RIGHT)
        loopsLTA.next_to(loopsLT2, RIGHT).scale(0.5)

        self.play(ReplacementTransform(loopsL2, loopsLT2), ReplacementTransform(loopsL3, loopsLT3), FadeIn(loopsLTA))

        self.wait(3)

        # consolidate loops and space
        self.play(ReplacementTransform(loopsL1, loopsU1), ReplacementTransform(loopsLT2, loopsU2), ReplacementTransform(loopsLT3, loopsU4), ReplacementTransform(spaceL, spaceU), ReplacementTransform(arrowL, arrowU),
                  FadeOut(loopsDLline), FadeOut(loopsLTA))
        #self.play(FadeOut(loopsL1), FadeOut(loopsL2), FadeOut(loopsL3), FadeOut(spaceL), FadeOut(arrowL))

        self.wait(3)


config["frame_rate"] = 20
