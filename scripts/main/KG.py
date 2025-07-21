import sys
import os
import networkx as nx
import json
import matplotlib.colors as mcolors
import random

try:
    from pyvis.network import Network
except ImportError:
    raise ImportError("pyvis is required for visualization. Please install it with 'pip install pyvis'.")

def generate_knowledge_graph(
    input_json_path: str,
    output_html_path: str = 'species_knowledge_graph.html',
    show_buttons: bool = True
) -> str:
    """
    Generate a knowledge graph from a JSON file and save as an HTML file.
    Args:
        input_json_path: Path to the processed relationships JSON file.
        output_html_path: Path to save the HTML visualization.
        show_buttons: Whether to show pyvis physics buttons.
    Returns:
        The path to the saved HTML file.
    """
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    G = nx.MultiDiGraph()
    relation_types = set()
    for doc in data:
        for rel in doc.get('relationships', []):
            relation_types.add(rel['relation'])
    color_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    random.shuffle(color_palette)
    relation2color = {rel: color_palette[i % len(color_palette)] for i, rel in enumerate(sorted(relation_types))}

    for doc in data:
        for rel in doc.get('relationships', []):
            src = rel['source_text']
            tgt = rel['target_text']
            label = rel['relation']
            conf = rel.get('confidence', 0)
            G.add_node(src, title=src)
            G.add_node(tgt, title=tgt)
            G.add_edge(src, tgt, label=label, color=relation2color[label], title=f"{label} (conf: {conf})")

    net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white', directed=True, cdn_resources='in_line')
    net.from_nx(G)

    for e in net.edges:
        rel = e['label']
        e['color'] = relation2color.get(rel, '#cccccc')
        e['title'] = rel
        e['width'] = 2

    for n in net.nodes:
        n['size'] = 15
        n['color'] = '#1f78b4'
        n['font'] = {'size': 18, 'color': 'white'}

    # Add legend for relation types as HTML (not displayed, but returned)
    legend_html = '<div style="padding:10px; background:#222; color:white; font-size:16px;">'
    legend_html += '<b>Relation Types:</b><br>'
    for rel, color in relation2color.items():
        legend_html += f'<span style="color:{color};">&#9632;</span> {rel} &nbsp; '
    legend_html += '</div>'

    if show_buttons:
        net.show_buttons(filter_=['physics'])
    net.save_graph(output_html_path)
    return output_html_path

if __name__ == "__main__":
    # Example usage
    input_json = "./s800_results/RE_FS_Results/speciesinteractions_llama3_3_t05_conf03_b5_final_version_of_prompt.json"
    output_html = "species_knowledge_graph.html"
    print(f"Generating knowledge graph from {input_json}...")
    out_path = generate_knowledge_graph(input_json, output_html)
    print(f"Knowledge graph saved as {out_path}")