import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def calculate_beam_deflection():
    reactions = 0 # collecting the reactions here
    supports = [] # and the supports
    
    # beam length
    while True:
        L_input = input("Enter the length of the beam (m): ")
        try:
            L = float(L_input)
            break 
        except ValueError:
            print("Error: Please enter a numeric value.")

    # number of supports
    while True:
        num_supports = input("Enter the number of supports: ")
        try:
            num_supports = int(num_supports)
            break 
        except ValueError:
            print("Error: Please enter a numeric value.")


    # support details
    support_positions = []
    for i in range(num_supports):
        print(f"Support {i + 1}:")
        while True:
            position = input("Enter the position along the beam (m) left to right: ")
            try:
                position = float(position)
                if position < 0 or position > L:
                    print("Error: Position must be between 0 and the beam length.")
                elif position in support_positions:  # Check for duplicates
                    print("Error: This position already has a support.")
                else:
                    support_positions.append(position)
                    break
            except ValueError:
                print("Error: Position must be a number.")
        
                # vlidate support type and add reaction
        while True:
            support_type = input("Enter the type of support (pinned, roller, or fixed): ").lower()
            if support_type == "pinned":
                reactions += 2 # vertical nda horizontal
                supports.append({"type": "pinned", "position": position})
                break
            elif support_type == "roller":
                reactions += 1  # vertical reaction only
                supports.append({"type": "roller", "position": position})
                break
            elif support_type == "fixed":
                reactions += 3  # vertical, horizontal, and moment reaction
                supports.append({"type": "fixed", "position": position})
                break
            else:
                print("Error: Invalid support type. Please enter pinned, roller, or fixed.")
                
    # number of loads
    while True:
        load_count = input("Enter the number of loads: ")
        try:
            load_count = int(load_count)
            break 
        except ValueError:
            print("Error: Please enter a numeric value.")
    
    # load details
    loads = [] #collecting loads here
    for i in range(load_count):
        print(f"Load {i + 1}:") #show the number of load to be customized
        while True:
            load_type = input("Enter the type of load (point, distributed, or moment): ").lower()
            if load_type == "point":
                magnitude = float(input("Enter point load magnitude (N): "))
                position = input("Enter the position along the beam (m) left to right: ")
                try:
                    position = float(position)
                    if position < 0 or position > L:
                        print("Error: Position must be between 0 and the beam length.")
                    elif position in loads:  # Check for duplicates
                        print("Error: This position already has a load.")
                    else:
                        loads.append({"type": "point", "position": position,"magnitude": magnitude})
                    break
                except ValueError:
                    print("Error: Position must be a number.")
            elif load_type == "distributed":
                magnitude = float(input("Enter distributed load magnitude (N/m): "))
                start = float(input("Enter the start position along the beam (m) left to right: "))
                end = float(input("Enter the end position along the beam (m) left to right: "))
                position = float(start) + (float(end) - float(start))/2 
                loads.append({"type": "distributed", "start": start, "end": end, "magnitude": magnitude})
                try:
                    start = float(start)
                    end = float(end)
                    if start < 0 or end > L or start >= end: #out of range input handling
                        print("Error: Start and end positions must be between 0 and the beam length, and start must be less than end.")
                    else:
                        break
                except ValueError:
                    print("Error: Positions must be numbers.")
            elif load_type == "moment":
                magnitude = float(input("Enter moment magnitude (Nm): "))
                position = input("Enter the position along the beam (m) left to right: ")
                loads.append({"type": "moment", "magnitude": magnitude, "position": position})  
                break       
        
    
    #check for determinacy

    if reactions == 3:
        print("The beam is statically determinate.")
    else:
        print("The beam is not statically determinate.")
        exit()


    # EI input
    while True:
        is_constantEI = input("Is the beam's EI constant? (y/n): ")
        try:
            if is_constantEI == 'y':
                while True:
                    E_input = input("Enter the value of E (Pa): ")
                    I_input = input("Enter the value of I (m^4): ")
                    try:
                        E_val = float(E_input)
                        I_val = float(I_input)
                        break
                    except ValueError:
                        print("Error: Please enter a numeric value.")
            elif is_constantEI == 'n':
                while True:
                    materialtype = input("""Enter the material type:   
                                         steel(s)
                                         aluminium(a)
                                         concrete(c)
                                         wood(w)
                                         plastic(p)
                                         rubber(r) : """).lower()
                    try:
                        if materialtype == 's':
                            E_val = 200e9
                        elif materialtype == 'a':
                            E_val = 70e9
                        elif materialtype == 'c':
                            E_val = 30e9
                        elif materialtype == 'w':
                            E_val = 10e9
                        elif materialtype == 'p':
                            E_val = 5e9
                        elif materialtype == 'r':
                            E_val = 0.1e9
                        else:
                            raise ValueError
                        break
                    except ValueError:
                        print("Error: Please enter a valid material type from the option.")
                        
                # geometry input
                while True:
                    Geometry_input = input("""Enter the geometry of the beam: 
                               Rectangular(R)
                               Circular(C)
                               Symmetric I-beam(I)
                               Hollow rectangle(Hr)
                               Hollow circular(Hc)
                               Triangular(t)
                               T-section(T) :""")
                    try:
                        if Geometry_input == 'R':
                            while True:
                                b_input = input("Enter the width of the beam (m): ")
                                h_input = input("Enter the height of the beam (m): ")
                                try:
                                    b = float(b_input)
                                    h = float(h_input)
                                    I_val = b*h**3/12
                                    break
                                except ValueError:
                                    print("Error: Please enter a numeric value.")
                        elif Geometry_input == 'I':
                            while True:
                                b_input = input("Enter the width of the flanges (m): ")
                                h_input = input("Enter the height of the beam (m): ")
                                tf_input = input("Enter the thickness of the flange (m): ")
                                tw_input = input("Enter the thickness of the web (m): ")
                                try:
                                    b = float(b_input)
                                    h = float(h_input)
                                    tf = float(tf_input)
                                    tw = float(tw_input)
                                    I_val = (1/12 * b * h**3) - 2 * (1/12 * ((b - tw)/2) * (h - 2*tf)**3)
                                    break
                                except ValueError:
                                    print("Error: Please enter a numeric value.")
                        elif Geometry_input == 'C':
                            while True:
                                r_input = input("Enter the radius of the beam (m): ")
                                try:
                                    r = float(r_input)
                                    I_val = np.pi*r**4/4
                                    break
                                except ValueError:
                                    print("Error: Please enter a numeric value.")
                        elif Geometry_input == 'Hc':
                            while True:
                                ro_input = input("Enter the outer radius of the beam (m): ")
                                ri_input = input("Enter the inner radius of the beam (m): ")
                                try:
                                    ro = float(ro_input)
                                    ri = float(ri_input)
                                    I_val = np.pi*(ro**4 - ri**4)/4
                                    break
                                except ValueError:
                                    print("Error: Please enter a numeric value.")
                        elif Geometry_input == 'Hr':
                            while True:
                                b_input = input("Enter the inner width of the beam (m): ")
                                h_input = input("Enter the inner height of the beam (m): ")
                                bo_input = input("Enter the outer width of the beam (m): ")
                                ho_input = input("Enter the outer height of the beam (m): ")
                                try:
                                    b = float(b_input)
                                    h = float(h_input)
                                    bo = float(bo_input)
                                    ho = float(ho_input)
                                    I_val = (bo*ho**3 - b*h**3)/12
                                    break
                                except ValueError:
                                    print("Error: Please enter a numeric value.")
                        elif Geometry_input == 't':
                            while True:
                                b_input = input("Enter the base width of the beam (m): ")
                                h_input = input("Enter the height of the beam (m): ")
                                try:
                                    b = float(b_input)
                                    h = float(h_input)
                                    I_val = b*h**3/36
                                    break
                                except ValueError:
                                    print("Error: Please enter a numeric value.")
                        elif Geometry_input == 'T':
                            while True:
                                b_input = input("Enter the width of the flange (m): ")
                                h_input = input("Enter the height of the beam (m): ")
                                tf_input = input("Enter the thickness of the flange (m): ")
                                tw_input = input("Enter the thickness of the web (m): ")
                                try:
                                    b = float(b_input)
                                    h = float(h_input)
                                    tf = float(tf_input)
                                    tw = float(tw_input)
                                    A1 = b * tf  # area of the flange
                                    A2 = tw * (h - tf)  # area of the web
                                    Y_1 = h - (tf / 2)
                                    Y_2 = h - tf - ((h - tf) / 2)
                                    Y_ = ((A1 * Y_1) + (A2 * Y_2)) / (A1 + A2)
                                    I_val = (1 / 12) * (b * tf**3) + A1 * ((h - Y_ - tf / 2)**2) + (1 / 12) * (tw * (h - tf)**3) + A2 * ((Y_ - (h - tf) / 2)**2)
                                    break
                                except ValueError:
                                    print("Error: Please enter a numeric value.")
                        else:
                            raise ValueError
                        break
                    except ValueError:
                        print("Error: Please enter a valid geometry from the option.")
                
            else:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter 'y' or 'n'.")
            
    # creating symbolic variables
    x = sp.Symbol('x')
    E, I = sp.symbols('E I')
    
    # moment equation
    M_total = 0
    vertical_reactions = []
    moment_reactions = []
    reaction_symbols = []

    # add reactions to the moment equation
    for i, support in enumerate(supports):
        pos = support['position']
        if support['type'] in ['pinned', 'roller']:
            R = sp.Symbol(f'R{i}')
            vertical_reactions.append(R)
            reaction_symbols.append(R)
            M_total += R * sp.SingularityFunction(x, pos, 1)
        elif support['type'] == 'fixed':
            R = sp.Symbol(f'R{i}')
            M = sp.Symbol(f'M{i}')
            vertical_reactions.append(R)
            moment_reactions.append(M)
            reaction_symbols.extend([R, M])
            M_total += R * sp.SingularityFunction(x, pos, 1) + M * sp.SingularityFunction(x, pos, 0)

    # add loads to the moment equation
    for load in loads:
        if load['type'] == 'point':
            M_total -= load['magnitude'] * sp.SingularityFunction(x, load['position'], 1)
        elif load['type'] == 'distributed':
            w, a, b = load['magnitude'], load['start'], load['end']
            M_total -= (w / 2) * (sp.SingularityFunction(x, a, 2) - sp.SingularityFunction(x, b, 2))
        elif load['type'] == 'moment':
            M_total -= load['magnitude'] * sp.SingularityFunction(x, load['position'], 0)

    # equilibrium equations
    total_point = sum(ld['magnitude'] for ld in loads if ld['type'] == 'point')
    total_distributed = sum(ld['magnitude'] * (ld['end'] - ld['start']) for ld in loads if ld['type'] == 'distributed')
    sum_Fy = sp.Eq(sum(vertical_reactions), total_point + total_distributed)
    
    # moment equilibrium about x=0 (external moments)
    sum_M = 0
    for i, support in enumerate(supports):
        pos = support['position']
        if support['type'] in ['pinned', 'roller']:
            sum_M += vertical_reactions[i] * pos
        elif support['type'] == 'fixed':
            sum_M += vertical_reactions[i] * pos + moment_reactions[i]
    for load in loads:
        if load['type'] == 'point':
            sum_M -= load['magnitude'] * load['position']
        elif load['type'] == 'distributed':
            w, a, b = load['magnitude'], load['start'], load['end']
            sum_M -= w * (b - a) * (a + b) / 2
        elif load['type'] == 'moment':
            sum_M -= load['magnitude']
    sum_M_eq = sp.Eq(sum_M, 0)

    # solve for reactions
    sol = sp.solve([sum_Fy, sum_M_eq], reaction_symbols)
    if not sol:
        raise ValueError("Statically indeterminate or no solution.")
    M_total = M_total.subs(sol)

    # integrate to find slope and deflection
    M_div_EI = M_total / (E * I) 
    theta = sp.integrate(M_div_EI, x) + sp.Symbol('C1') # we got slope here
    y = sp.integrate(theta, x) + sp.Symbol('C2') # and defelection here

    # boundary conditions
    bc_eq = []
    for support in supports:
        pos = support['position']
        bc_eq.append(sp.Eq(y.subs(x, pos), 0)) # deflection zero for the supports
        if support['type'] == 'fixed':
            bc_eq.append(sp.Eq(theta.subs(x, pos), 0)) # zero slope for fixed of course
    
    constants = sp.solve(bc_eq, ['C1', 'C2'])
    y_deflection = y.subs(constants).subs({E: E_val, I: I_val})
    theta_slope = theta.subs(constants).subs({E: E_val, I: I_val})

    # plot
    y_lambda = sp.lambdify(x, y_deflection, 'numpy')
    x_vals = np.linspace(0, L, 100)
    y_vals = y_lambda(x_vals)


    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, y_vals, '--', color='gray', label='Elastic Curve (Deflection)')
    plt.title('Beam Deflection')
    plt.xlabel('Position (m)')
    plt.ylabel('Deflection (m)')
    plt.axhline(0, color='black', lw=2, label='Original Beam Line')
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_deflection, theta_slope

y_eq, theta_eq = calculate_beam_deflection()
print("Deflection equation:")
sp.pprint(y_eq)
print("\nRotation equation (radians):")
sp.pprint(theta_eq)