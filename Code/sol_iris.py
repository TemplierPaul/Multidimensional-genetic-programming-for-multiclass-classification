data = iris
print(data.columns)

# Create the Var objects
sep_l = Var('sep_l')
pet_l = Var('pet_l')
pet_w = Var('pet_w')

# The operations return Var objects
a = sep_l + pet_l
print(a)
b = pet_l * pet_w
print(b)
c = sin(exp(pet_l))
print(c)

# Transforming data 
new_data = pd.DataFrame()
new_data[str(a)] = a.compute(data)
new_data[str(b)] = b.compute(data)
new_data[str(c)] = c.compute(data)
new_data['target'] = data['target']

# Plotting data
g = sns.pairplot(new_data, vars=new_data.columns[:-1], hue='target')
g.fig.set_size_inches(10,10)
