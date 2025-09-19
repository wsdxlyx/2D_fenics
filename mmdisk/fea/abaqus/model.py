# -*- coding: utf-8 -*-
"""
AUTHOR: SHEN WANG @ LEHIGH UNIVERSITY

"""

import math

import assembly
import interaction
import job
import load
import mesh
import numpy as np
import sketch
import step
import visualization
from abaqus import *
from abaqusConstants import *

mat_format = "Mat%i"


def create_constant_thickness_1d_model(
    cae_path, r1, r2, H, n_sec, n_secele, model_name, part_name, plastic=True
):
    instance_name = part_name + "-1"
    L = r2 - r1  # LENGTH; M
    l = 1.0 * L / n_sec  # LENGTH OF EACH SECTION
    sec_center_pos = [
        l / 2.0 + i * l for i in range(0, n_sec)
    ]  # DISTANCE FROM EACH SECTION CENTER TO THE INNER SURFACE
    sec_center_x = [
        r1 + i for i in sec_center_pos
    ]  # X-COORDINATES OF EACH SECTION CENTER
    mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)
    m = mdb.models[model_name]
    a = m.rootAssembly
    vp = session.viewports["Viewport: 1"]

    # SET DISPLAY
    vp.setValues(displayedObject=a)
    vp.assemblyDisplay.setValues(step="Initial")
    vp.partDisplay.setValues(mesh=OFF)
    vp.partDisplay.meshOptions.setValues(meshTechnique=OFF)
    vp.partDisplay.geometryOptions.setValues(referenceRepresentation=ON)
    vp.setValues(displayedObject=None)

    # ------------------------------------------------
    # CREATE A RECTANGLE 2D PART BY DRAWING A SKETCH
    # ------------------------------------------------
    # SET SKETCH OBJECTS
    s = m.ConstrainedSketch(name="__profile__", sheetSize=1.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.sketchOptions.setValues(viewStyle=AXISYM)
    s.setPrimaryObject(option=STANDALONE)
    s.ConstructionLine(point1=(0.0, 0.0), point2=(0.0, 1.0))  # AXIS
    s.FixedConstraint(entity=g[2])

    # BUILD FIXED CONSTRUCTION LINES FOR CARTISAN COORDINATES
    s.ConstructionLine(point1=(0.0, 0.0), point2=(1.0, 0.0))
    s.FixedConstraint(entity=g[3])  # HORIZONTAL LINE

    # DRAW RECTANGEL AND ADD CONSTRAINS AND DIMENSIONS
    s.rectangle(point1=(0.01, 0.05), point2=(0.05, 0))
    s.CoincidentConstraint(entity1=v[1], entity2=g[3], addUndoState=False)
    s.DistanceDimension(
        entity1=g[2], entity2=g[4], textPoint=(r1 / 2, 1.1 * H), value=r1
    )
    s.DistanceDimension(
        entity1=g[2], entity2=g[6], textPoint=(r2 / 2, 1.2 * H), value=r2
    )
    s.DistanceDimension(
        entity1=g[5], entity2=g[7], textPoint=(1.1 * r2, 1.2 * H), value=H
    )

    # CREATE A PART BASED ON THE SKETCH
    p = m.Part(name=part_name, dimensionality=AXISYMMETRIC, type=DEFORMABLE_BODY)
    p = m.parts[part_name]
    p.BaseShell(sketch=s)
    s.unsetPrimaryObject()

    vp.setValues(displayedObject=p)
    del m.sketches["__profile__"]

    f, e, d = p.faces, p.edges, p.datums
    t = p.MakeSketchTransform(
        sketchPlane=f[0], sketchPlaneSide=SIDE1, origin=(0.0, 0.0, 0.0)
    )
    s = mdb.models[model_name].ConstrainedSketch(
        name="__profile__", sheetSize=0.41, gridSpacing=0.01, transform=t
    )
    g, v, d1, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=SUPERIMPOSE)
    p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)

    # DRAW VERTICAL LINES TO DIVIDE THE RECTANGLE INTO N SECTIONS
    l = 1.0 * L / n_sec
    line_xcord = [r1 + i * l for i in range(1, n_sec)]
    for i in line_xcord:
        s.Line(point1=(i, H), point2=(i, 0))

    del i, line_xcord

    # CREATE PARTITIONS
    p.PartitionFaceBySketch(faces=f.findAt(((r1 + L / 2.0, H / 2.0, 0),)), sketch=s)
    s.unsetPrimaryObject()
    del m.sketches["__profile__"]

    # -----------------------------------
    # DEFINE MATERIAL PROPERTIES AND SET
    # -----------------------------------
    p.Set(edges=e.findAt(((r1, H / 2.0, 0),)), name="Left")
    p.Set(edges=e.findAt(((r2, 0, 0),)), name="Right")
    p.Set(
        edges=e.getByBoundingBox(
            xMin=0 - 1.0 * L,
            yMin=-1.0 * H,
            zMin=0.0,
            xMax=1.1 * r2,
            yMax=0.5 * H,
            zMax=0.0,
        ),
        name="Bottom",
    )
    p.Set(
        faces=f.getByBoundingBox(
            xMin=0 - 1.0 * L,
            yMin=-1.0 * H,
            zMin=0.0,
            xMax=1.1 * r2,
            yMax=2.0 * H,
            zMax=0.0,
        ),
        name="All",
    )

    for i in range(n_sec):
        # DEFINE MATERIAL PROPERTY
        mat_name = mat_format % (i + 1)
        m.Material(name=mat_name)
        m.materials[mat_name].Elastic(table=((210e9, 0.3),))
        if plastic:
            m.materials[mat_name].Plastic(table=((800e6, 0.0),))
        m.materials[mat_name].Expansion(table=((1.4e-5,),))
        m.materials[mat_name].Conductivity(table=((21.9,),))
        m.materials[mat_name].Density(table=((8000.0,),))
        m.materials[mat_name].SpecificHeat(table=((436.0,),))

        # CREATE MATERIAL SECTION
        sec_name = "Sec" + str(i + 1)
        p.Set(faces=f.findAt(((sec_center_x[i], 0, 0),)), name=sec_name)
        m.HomogeneousSolidSection(name=sec_name, material=mat_name, thickness=None)

        # ASSIGN MATERIAL TO SECTION
        p.SectionAssignment(
            region=p.sets[sec_name],
            sectionName=sec_name,
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )

    # ----------------------------
    # CREATE INSTANCE
    # ----------------------------
    vp.setValues(displayedObject=a)

    a.DatumCsysByDefault(CARTESIAN)
    a.Instance(name=instance_name, part=p, dependent=ON)
    inst = a.instances[instance_name]

    vp.setValues(displayedObject=p)
    pickedRegions = f.getByBoundingBox(
        xMin=0 - 1.0 * L,
        yMin=-1.0 * H,
        zMin=0.0,
        xMax=1.1 * r2,
        yMax=2.0 * H,
        zMax=0.0,
    )
    p.setMeshControls(regions=pickedRegions, elemShape=QUAD)
    elemType1 = mesh.ElemType(elemCode=CAX4RT, elemLibrary=STANDARD)
    elemType2 = mesh.ElemType(elemCode=CAX3RT, elemLibrary=STANDARD)
    p.setElementType(regions=(pickedRegions,), elemTypes=(elemType1, elemType2))
    p.seedPart(size=1.0 * l / n_secele, deviationFactor=0.1, minSizeFactor=0.1)

    # Mesh Control

    ratio = 0.002 / 0.0005
    n_biased = 7

    # BIASED SEEDS NEAR AT SHAFT
    TopEdge = e.findAt(((sec_center_x[0], H, 0),))
    BotEdge = e.findAt(((sec_center_x[0], 0, 0),))
    p.seedEdgeByBias(
        biasMethod=SINGLE,
        end1Edges=BotEdge,
        end2Edges=TopEdge,
        ratio=ratio,
        number=n_biased,
        constraint=FINER,
    )

    # p.seedEdgeByNumber(
    #     edges=e.findAt(
    #         (
    #             (sec_center_x[1], H, 0),
    #             (sec_center_x[1], 0, 0),
    #         ),
    #         (
    #             (sec_center_x[-2], H, 0),
    #             (sec_center_x[-2], 0, 0),
    #         ),
    #     ),
    #     number=2,
    #     constraint=FINER,
    # )

    # BIASED SEEDS NEAR AT RIM
    TopEdge = e.findAt(((sec_center_x[-1], H, 0),))
    BotEdge = e.findAt(((sec_center_x[-1], 0, 0),))
    p.seedEdgeByBias(
        biasMethod=SINGLE,
        end1Edges=TopEdge,
        end2Edges=BotEdge,
        ratio=ratio,
        number=n_biased,
        constraint=FINER,
    )

    # GENERATE MESH
    p.generateMesh()
    a.regenerate()
    vp.setValues(displayedObject=a)

    # Mechanical BCs
    m.YsymmBC(
        name="Y_symm",
        createStepName="Initial",
        region=inst.sets["Bottom"],
        localCsys=None,
    )

    # LEFT BOUNDARY
    m.DisplacementBC(
        name="Left_radial",
        createStepName="Initial",
        region=inst.sets["Left"],
        u1=SET,
        u2=UNSET,
        ur3=SET,
        amplitude=UNSET,
        distributionType=UNIFORM,
        fieldName="",
        localCsys=None,
    )

    mdb.saveAs(pathName=cae_path)


def configure_coupled_case(
    model_name,
    n_cycle,
    ini_temp,
    left_temp,
    right_temp,
    omega_rad,
    Onecycle,
    mat_prop=None,
    plastic=True,
):
    """Configure the case for the Abaqus simulation.

    Assumes the abaqus model has been loaded.

    Parameters
    ----------
    n_cycle : int
        Number of temperature cycles.
    ini_temp : float
        Initial temperature.
    left_temp : float
        Constant hub temperature.
    right_temp : float
        Maximum rim temperature.
    omega_rad : float
        Angular velocity in rad/s.
    Onecycle : np.ndarray[float]
        Time spent at each temperature in one cycle.
    mat_prop : np.ndarray[float]
        Material properties. Each row corresponds to a property and each column to an element.
        Order: E, nu, Sy, alpha, k, rho.
    """
    T_min = left_temp  # HIGHEST TEMPERATURE DURING CYCLING; KELVIN
    T_max = right_temp  # LOWEST TEMPERATURE DURING CYCLING; KELVIN

    if len(Onecycle) != 4:
        raise ValueError("Onecycle must have 4 elements")

    instance_name = model_name + "-1"

    m = mdb.models[model_name]
    a = m.rootAssembly
    inst = a.instances[instance_name]

    if mat_prop is not None:
        # Configure material
        for i in range(len(m.materials)):
            mat_name = mat_format % (i + 1)
            m.materials[mat_name].Elastic(table=((mat_prop[0, i], mat_prop[1, i]),))
            if plastic:
                m.materials[mat_name].Plastic(table=((mat_prop[2, i], 0.0),))
            m.materials[mat_name].Expansion(table=((mat_prop[3, i],),))
            m.materials[mat_name].Conductivity(table=((mat_prop[4, i],),))
            m.materials[mat_name].Density(table=((mat_prop[5, i],),))
            m.materials[mat_name].SpecificHeat(table=((mat_prop[6, i],),))

    # Configure step
    period = Onecycle.sum()
    Totalcycletimepoint = np.zeros(n_cycle * len(Onecycle) + 1)
    Totalcycletimepoint[1:] = np.tile(Onecycle, n_cycle).cumsum()

    one_cycle_output = [
        Onecycle[0] / 2.0,
        Onecycle[0] / 2.0,
        Onecycle[1] / 2.0,
        Onecycle[1] / 2.0,
        Onecycle[2] / 2.0,
        Onecycle[2] / 2.0,
        Onecycle[3],
    ]
    OutputTimePoints = np.zeros(n_cycle * len(one_cycle_output) + 1)
    OutputTimePoints[1:] = np.tile(one_cycle_output, n_cycle).cumsum()

    m.CoupledTempDisplacementStep(
        name="StepAll",
        previous="Initial",
        timePeriod=Totalcycletimepoint[-1],
        initialInc=0.01,
        minInc=1e-08,
        maxInc=period,
        maxNumInc=100000,
        deltmx=10000,
        nlgeom=ON,
    )

    OutputTimePoints_tuple = [(round(i, 4),) for i in OutputTimePoints]
    m.TimePoint(name="OutputTimePoints", points=OutputTimePoints_tuple)

    output_fields = ["S", "MISES", "E", "U", "V", "NT"]
    if plastic:
        output_fields.append("PEEQ")
    m.fieldOutputRequests["F-Output-1"].setValues(
        variables=output_fields,
        timePoint="OutputTimePoints",
    )

    # ----------------------------
    # DEFINE BOUNDARY CONDITIONS
    # ----------------------------
    ## DEFINE LOAD AMPLITUDE
    mapping_dic = {0: T_min, 1: T_max, 2: T_max, 3: T_min}
    time_amp = [
        (Totalcycletimepoint[i], mapping_dic[i % 4])
        for i in range(len(Totalcycletimepoint))
    ]
    m.TabularAmplitude(
        name="Temp_Cycling", timeSpan=STEP, smooth=SOLVER_DEFAULT, data=time_amp
    )

    # m.TabularAmplitude(name='Amp_BeforeCycling', timeSpan=STEP, smooth=SOLVER_DEFAULT,
    #                   data=((0.0, 0.0), (1.0, 1.0), (1 + n_cycle, 1.0)))

    # CENTRIFUGAL LOAD BY DEFINING ROTATIONAL SPEED
    m.RotationalBodyForce(
        name="Centrifugal",
        createStepName="StepAll",
        region=inst.sets["All"],
        magnitude=omega_rad,
        centrifugal=ON,
        rotaryAcceleration=OFF,
        point1=(0.0, 0.0, 0.0),
        point2=(0.0, 1.0, 0.0),
    )

    ## THERMAL BCS
    # INITIAL TEMPERATURE FIELD
    m.Temperature(
        name="InitialTempField",
        createStepName="Initial",
        region=inst.sets["All"],
        distributionType=UNIFORM,
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
        magnitudes=(ini_temp,),
    )

    # THERMAL CONVECTION
    # for i in range(n_sec):
    #    m.FilmCondition(name='ThermConv'+str(i+1), createStepName='Mean',
    #                    surface = inst.surfaces['Surf'+str(i+1)], definition=EMBEDDED_COEFF,
    #                    filmCoeff = MP_hconv[i], filmCoeffAmplitude='', sinkTemperature = sink_temp,
    #                    sinkAmplitude='', sinkDistributionType=UNIFORM, sinkFieldName='')

    # TEMPERATURE AT LEFT AND RIGHT BOUNDARIES
    m.TemperatureBC(
        name="Temp_Left",
        createStepName="StepAll",
        region=inst.sets["Left"],
        distributionType=UNIFORM,
        fieldName="",
        magnitude=0.0,
    )

    m.TemperatureBC(
        name="Temp_Right",
        createStepName="StepAll",
        region=inst.sets["Right"],
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName="",
        magnitude=1,
        amplitude="Temp_Cycling",
    )

    m.TemperatureBC(
        name="Temp_Left",
        createStepName="StepAll",
        region=inst.sets["Left"],
        distributionType=UNIFORM,
        fieldName="",
        magnitude=left_temp,
    )

    m.TemperatureBC(
        name="Temp_Right",
        createStepName="StepAll",
        region=inst.sets["Right"],
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName="",
        magnitude=1,
        amplitude="Temp_Cycling",
    )


def configure_static_case(
    model_name,
    omega_rad,
    mat_prop=None,
    plastic=True,
):
    instance_name = model_name + "-1"

    m = mdb.models[model_name]
    a = m.rootAssembly
    inst = a.instances[instance_name]

    if mat_prop is not None:
        # Configure material
        for i in range(len(m.materials)):
            mat_name = mat_format % (i + 1)
            m.materials[mat_name].Elastic(table=((mat_prop[0, i], mat_prop[1, i]),))
            if plastic:
                m.materials[mat_name].Plastic(table=((mat_prop[2, i], 0.0),))
            m.materials[mat_name].Expansion(table=((mat_prop[3, i],),))
            m.materials[mat_name].Conductivity(table=((mat_prop[4, i],),))
            m.materials[mat_name].Density(table=((mat_prop[5, i],),))
            m.materials[mat_name].SpecificHeat(table=((mat_prop[6, i],),))

    m.StaticStep(name="Step-1", previous="Initial", nlgeom=ON, maxInc=0.01)
    # ----------------------------
    # DEFINE BOUNDARY CONDITIONS
    # ----------------------------
    # CENTRIFUGAL LOAD BY DEFINING ROTATIONAL SPEED
    m.RotationalBodyForce(
        name="Centrifugal",
        createStepName="Step-1",
        region=inst.sets["All"],
        magnitude=omega_rad,
        centrifugal=ON,
        rotaryAcceleration=OFF,
        point1=(0.0, 0.0, 0.0),
        point2=(0.0, 1.0, 0.0),
    )


if __name__ == "__main__":
    from caeModules import *
    from driverUtils import executeOnCaeStartup

    executeOnCaeStartup()
    Mdb()

    model_name = "Disk-Universal"

    create_constant_thickness_1d_model(
        "disk-universal.cae",
        0.0,
        0.18,
        0.001,
        100,
        4,
        model_name,
        model_name,
        plastic=True,
    )

    configure_static_case(model_name, 100.0, plastic=True)
    job_name = model_name + "-Static"
    job = mdb.Job(name=job_name, model=model_name)
    job.writeInput()

    openMdb("disk-universal.cae")
    configure_coupled_case(
        model_name,
        10,
        0.0,
        0.0,
        100.0,
        100.0,
        np.array([1.0, 180.0, 3.0, 1500.0]),
        plastic=True,
    )
    job_name = model_name
    job = mdb.Job(name=job_name, model=model_name)
    job.writeInput()

    mdb.save()
