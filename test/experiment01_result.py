import matplotlib.pyplot as plt
import numpy as np

def test1():
    boundary_list = [
        [ 0,  67, 101, 147, 182, 221, 250, 254],
        [ 0,  67,  93, 155, 172, 230, 251, 254],
        [ 0,  67,  96, 155, 171, 225, 251, 254],
        [ 0,  68,  95, 155, 171, 224, 252, 254],
        [ 0,  68,  97, 152, 171, 228, 252, 254]
    ]
 
    # boundary_array = np.array(boundary_list)
    # diff_list = []
    # ave_list = []
    # for i in range(len(boundary_array)):
    #     data = boundary_array[i,:]
    #     ave = np.mean(data)
    #     ave_list.append(ave)
    
    # ave_array = np.array(ave_list)

    # for i in range(boundary_array.shape[0]):
    #     diff = 0
    #     for j in range(boundary_array.shape[1]):
    #         diff += (boundary_array[i,j] - ave_array[i])**2
    #     diff_list.append(np.sqrt(diff)/boundary_array.shape[1])
    
    # diff_array = np.array(diff_list)
    
    # print(diff_array)
    
    # plt.plot(diff_array)
    # plt.title("Resolution Effect on Boundary Value")
    # plt.xlabel("Iteration")
    # plt.ylabel("Normalized Boundary Value Difference")
    # plt.xticks(np.arange(0, 5))
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.savefig("img/resolution_effect_on_boundary_value.png", dpi=600)
    # plt.show()

    boundary_list = np.array(boundary_list)
    # representive_list = np.array(representive_list)

    # Choose a suitable color palette for boundary points (e.g., tab10)
    colors_boundary = plt.cm.tab10.colors  # Tab10 palette for distinct colors

    datas = []
    # Plot boundary points on the first subplot
    for i in range(0, boundary_list.shape[1]):
        y = boundary_list[:,i]
        x = np.arange(1, len(y)+1)
        data = np.column_stack((x, y))
        datas.append(data)
        plt.plot(data[:,0],data[:,1], label=f"Iteration {i+1}", marker='o', markersize=6, linestyle='-', alpha=0.8, color=colors_boundary[i % len(colors_boundary)])
    plt.title("resolution effect on boundary isovalue", fontsize=14)
    plt.xlabel("resolution scale", fontsize=12)
    plt.ylabel("boundary isovalue", fontsize=12)
    plt.xticks(np.arange(1, 6))
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("img/resolution_effect_on_boundary_value.png", dpi=600)
    # plt.legend(title="boundary index", fontsize=10)

    plt.show()

    # # Generate a list of colors from the Blues colormap
    # colors_representive = plt.cm.Blues(np.linspace(0.2, 0.8, len(representive_list)))

    # Plot representative points on the second subplot
    # for i, representive in enumerate(representive_list):
    #     ax2.plot(representive, label=f"Iteration {i+1}", marker='o', linestyle='-', color=colors_representive[i])
    # ax2.set_title("Representative Points", fontsize=14)
    # ax2.set_xlabel("Index", fontsize=12)
    # ax2.set_ylabel("Representative Value", fontsize=12)
    # ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax2.legend(title="Iterations", fontsize=10)

    # # Tight layout to avoid overlap
    # plt.tight_layout()

    # # Show the plot
    

# Run the function to plot the results
test1()