try:
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
except ImportError as err:
    print(err)
    raise err


class QuadObj:
    def __init__(self):
        self.quadObj = 0

    def initQuadObj(self):
        if self.quadObj == 0:
            self.quadObj = glu.gluNewQuadric()

        if self.quadObj == 0:
            raise RuntimeError('cannot create quadObj')

    def predraw(self, solid):
        if self.quadObj == 0:
            self.quadObj = glu.gluNewQuadric()

        if self.quadObj == 0:
            raise RuntimeError('cannot create quadObj')
        glu.gluQuadricDrawStyle(self.quadObj, glu.GLU_FILL if solid else glu.GLU_LINE)
        glu.gluQuadricNormals(self.quadObj, glu.GLU_SMOOTH)
        ''' If we ever changed/used the texture or orientation state
        of quadObj, we’d need to change it to the defaults here
        with gluQuadricTexture and/or gluQuadricOrientation. '''


class Sphere(QuadObj):
    def __init__(self):
        super().__init__()

    def draw(self, solid, radius, slices, stacks):
        self.predraw(solid)
        glu.gluSphere(self.quadObj, radius, slices, stacks)

    def __call__(self, solid, radius, slices, stacks):
        self.draw(solid, radius, slices, stacks)


class Cone(QuadObj):
    def __init__(self):
        super().__init__()

    def draw(self, solid, base, height, slices, stacks):
        self.predraw(solid)
        glu.gluCylinder(self.quadObj, base, 0, height, slices, stacks)

    def __call__(self, solid, base, height, slices, stacks):
        self.draw(solid, base, height, slices, stacks)


drawSphere = Sphere()
drawCone = Cone()
