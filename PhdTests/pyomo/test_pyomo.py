from pyomo.environ import ConcreteModel, Var, Binary, Constraint, Objective, minimize, log, SolverFactory

#Create a simple model
model = ConcreteModel()

model.x = Var(bounds=(1.0,10.0),initialize=7.0)
model.y = Var(within=Binary)

model.c1 = Constraint(expr=(model.x-3.0)**2 <= 50.0*(1-model.y))
model.c2 = Constraint(expr=model.x*log(model.x)+5.0 <= 50.0*(model.y))
# issue 1187 --> the first master problem does not have linear contrainst on y hence a bug...
# Fix: Add a useless manual constraint
model.c3 = Constraint(expr=model.y<=1)  

model.objective = Objective(expr=model.x, sense=minimize)

#Solve the model using MindtPy
solver=SolverFactory('mindtpy')
# solver.CONFIG._data['iteration_limit']._data=100
solver.solve(model, mip_solver='glpk', nlp_solver='ipopt', tee=True)

model.objective.display()
model.display()
model.pprint()