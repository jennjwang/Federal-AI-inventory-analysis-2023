import numpy as np
import pandas as pd
import json
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

def load_data():
    embedding = np.load("data/GPT_umap.npy")
    df_keywords = pd.read_csv("data/GPT_cluster_keywords.csv")
    clusters = np.load("data/GPT_clusters.npy")

    f_json = "data/GPT_automated_analysis.json"

    with open(f_json) as FIN:
        js = json.load(FIN)
    df = pd.DataFrame(js["record_content"])
    df["ux"], df["uy"] = embedding.T

    return df, df_keywords, clusters

df, df_keywords, clusters = load_data()

# print(df[df['Department_Code'] == 'VA'])
# print(df[df['Department_Code'] == 'HHS'])

hhs = df[df['Department_Code'] == 'HHS']
va = df[df['Department_Code'] == 'DOE']

hhs_coors = list(zip(hhs['ux'], hhs['uy']))
va_coors = list(zip(va['ux'], va['uy']))

def plot_polygon_or_multipolygon(ax, geom, **kwargs):
    """Utility function to plot a Polygon or MultiPolygon."""
    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.fill(x, y, **kwargs)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, **kwargs)

def convex_hull_overlap(list1, list2):
    multi_point = MultiPoint(list1)
    
    # Compute the convex hull
    hull = multi_point.convex_hull

    fig, ax = plt.subplots()
    
    # Plot the original points
    x, y = zip(*list1)
    ax.scatter(x, y, c='blue', label='Original Points')
    
    # Plot the convex hull
    if hull.geom_type == "Polygon":
        x, y = hull.exterior.xy
        ax.plot(x, y, c='red', label='Convex Hull')
    elif hull.geom_type == "Point":
        ax.scatter(hull.x, hull.y, c='red', label='Convex Hull')
    elif hull.geom_type == "LineString":
        x, y = hull.xy
        ax.plot(x, y, c='red', label='Convex Hull')
    
    ax.legend()
    plt.show()

# area = convex_hull_overlap(hhs_coors, va_coors)
# print(area)

def svm_overlap(list1, list2):

    list1 = np.array(list1)
    list2 = np.array(list2)

    clfl1 = OneClassSVM(kernel="rbf", gamma=0.7, nu=0.05).fit(list1)
    clfl2 = OneClassSVM(kernel="rbf", gamma=1, nu=0.02).fit(list2)

    # Plotting
    fig, ax = plt.subplots()

    # Plot the original points
    ax.scatter(list1[:, 0], list1[:, 1], c='blue', label='HHS')

    # Create a grid to visualize the decision function
    xx, yy = np.meshgrid(np.linspace(min(list1[:, 0]) - 1, max(list1[:, 0]) + 1, 500),
                        np.linspace(min(list1[:, 1]) - 1, max(list1[:, 1]) + 1, 500))
    Z1 = clfl1.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    ax.scatter(list2[:, 0], list2[:, 1], c='red', label='VA')

    xx2, yy2 = np.meshgrid(np.linspace(min(list2[:, 0]) - 1, max(list2[:, 0]) + 1, 500),
                        np.linspace(min(list2[:, 1]) - 1, max(list2[:, 1]) + 1, 500))

    Z2 = clfl2.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z2 = Z2.reshape(xx.shape)

    # Plot the decision function and the boundary
    # ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 1), cmap=plt.cm.PuBu)
    ax.contourf(xx, yy, Z1, levels=[0, Z1.max()], colors='skyblue', alpha=0.5)
    ax.contourf(xx2, yy2, Z2, levels=[0, Z2.max()], colors='orange', alpha=0.5)

    ax.set_title('Overlapping Area Between HHS and VA')
    ax.legend()
    plt.show()

area = svm_overlap(hhs_coors, va_coors)
print(area)

def polygons_overlap(list1, list2):
    # Convert lists to polygons
    poly1 = Polygon(list1)
    poly2 = Polygon(list2)

    if not poly1.is_valid:
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        poly2 = poly2.buffer(0)
    
    intersecting_polygon = poly1.intersection(poly2)

    fig, ax = plt.subplots()
    
    # Plot the two polygons
    x1, y1 = zip(*list1)
    ax.scatter(x1, y1, color='red', marker='o', label='Points from list1')
    x2, y2 = zip(*list2)
    ax.scatter(x2, y2, color='blue', marker='x', label='Points from list2')
    
    plot_polygon_or_multipolygon(ax, poly1, alpha=0.5, fc='red', label='Polygon 1')
    plot_polygon_or_multipolygon(ax, poly2, alpha=0.5, fc='blue', label='Polygon 2')
    
    # Plot the overlapping area
    if intersecting_polygon.is_empty:  # Check if there's an overlap
        print("No overlapping area!")
    else:
        plot_polygon_or_multipolygon(ax, intersecting_polygon, alpha=0.7, fc='green', label='Overlapping Area')
    
    ax.set_title('Polygons and Overlapping Area')
    # ax.legend()
    plt.show()

# area = polygons_overlap(hhs_coors, va_coors)
# print(area)

# print(df["Department_Code"].unique())
# print(df.head())