import gmsh
import csv

def export_xyz_to_csv(msh_file, output_csv_file):
    gmsh.initialize()
    gmsh.open(msh_file)
    
    # Retrieve all nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    
    gmsh.finalize()

    # Write the coordinates to a CSV file
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['x', 'y', 'z'])  # Write the header
        for i in range(len(node_tags)):
            x = node_coords[3 * i]
            y = node_coords[3 * i + 1]
            z = node_coords[3 * i + 2]
            csv_writer.writerow([x, y, z])

if __name__=="__main__":
    msh_file = '/work1/Doutorado/data/3D/static/rocker-arm/rocker-arm.msh'
    output_csv_file = msh_file + '0.csv'

    export_xyz_to_csv(msh_file, output_csv_file)
    print(f"Vertices exported to {output_csv_file}")
