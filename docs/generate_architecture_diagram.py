#!/usr/bin/env python3
"""
Generate architecture diagram for BooFun library documentation.

This script creates a visual representation of the library's modular architecture
using matplotlib and saves it as both PNG and SVG formats.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, FancyBboxPatch


def create_architecture_diagram():
    """Create the BooFun architecture diagram."""

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Color scheme
    colors = {
        "api": "#e3f2fd",  # Light blue
        "core": "#fff3e0",  # Light orange
        "analysis": "#fce4ec",  # Light pink
        "repr": "#e8f5e8",  # Light green
        "ext": "#f1f8e9",  # Very light green
        "edge": "#666666",  # Dark gray for edges
    }

    # Helper function to create boxes
    def create_box(x, y, width, height, text, color, text_size=10):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor=colors["edge"],
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=text_size,
            fontweight="bold",
            wrap=True,
        )
        return box

    # Helper function to create connections
    def create_connection(box1_x, box1_y, box1_w, box1_h, box2_x, box2_y, box2_w, box2_h):
        # Connect bottom of box1 to top of box2
        start_x = box1_x + box1_w / 2
        start_y = box1_y
        end_x = box2_x + box2_w / 2
        end_y = box2_y + box2_h

        ax.arrow(
            start_x,
            start_y,
            end_x - start_x,
            end_y - start_y,
            head_width=0.08,
            head_length=0.08,
            fc=colors["edge"],
            ec=colors["edge"],
            alpha=0.7,
        )

    # Title
    ax.text(
        5,
        9.5,
        "BooFun Library Architecture",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    # Layer 1: User Interface
    create_box(4, 8.5, 2, 0.8, "User Interface\n(Python API)", colors["api"], 11)

    # Layer 2: API Layer
    create_box(4, 7.2, 2, 0.8, "API Layer\n(bf.create, bf.*)", colors["api"], 10)

    # Layer 3: Core Engine
    create_box(3.5, 5.8, 3, 0.8, "Core Engine\n(BooleanFunction, Factory)", colors["core"], 11)

    # Layer 4: Main Components
    # Boolean Function Core
    create_box(
        0.5,
        4.2,
        1.8,
        1,
        "Boolean Function\n• Evaluation\n• Conversion\n• Caching",
        colors["core"],
        9,
    )

    # Representations
    create_box(
        2.8,
        4.2,
        1.8,
        1,
        "Representations\n• Truth Table\n• ANF • Fourier\n• BDD • Circuit",
        colors["repr"],
        8,
    )

    # Analysis Tools
    create_box(
        5.1,
        4.2,
        1.8,
        1,
        "Analysis Tools\n• Spectral\n• Property Testing\n• Influences",
        colors["analysis"],
        9,
    )

    # Extensions
    create_box(
        7.4, 4.2, 1.8, 1, "Extensions\n• Quantum Complexity\n• Visualization\n• Adapters", colors["ext"], 9
    )

    # Layer 5: Detailed Components
    # Performance layer
    create_box(0.2, 2.5, 1.2, 0.8, "Batch\nProcessing", colors["core"], 9)
    create_box(1.6, 2.5, 1.2, 0.8, "GPU\nAcceleration", colors["core"], 9)

    # Conversion system
    create_box(3.1, 2.5, 1.2, 0.8, "Conversion\nGraph", colors["repr"], 9)
    create_box(4.5, 2.5, 1.2, 0.8, "Registry\nSystem", colors["repr"], 9)

    # Analysis details
    create_box(6.0, 2.5, 1.2, 0.8, "Fourier\nAnalysis", colors["analysis"], 9)
    create_box(7.4, 2.5, 1.2, 0.8, "Property\nTesting", colors["analysis"], 9)

    # Bottom layer: Built-ins and utilities
    create_box(1, 1, 1.5, 0.8, "Built-in\nFunctions", colors["repr"], 9)
    create_box(3, 1, 1.5, 0.8, "Error\nModels", colors["core"], 9)
    create_box(5, 1, 1.5, 0.8, "Testing\nFramework", colors["ext"], 9)
    create_box(7, 1, 1.5, 0.8, "Utilities\n& Helpers", colors["ext"], 9)

    # Add some connecting arrows (simplified)
    # User Interface to API
    create_connection(4, 8.5, 2, 0.8, 4, 7.2, 2, 0.8)

    # API to Core Engine
    create_connection(4, 7.2, 2, 0.8, 3.5, 5.8, 3, 0.8)

    # Core Engine to main components
    for x_pos in [1.4, 3.7, 6.0, 8.3]:
        ax.arrow(
            5,
            5.8,
            x_pos - 5,
            -0.4,
            head_width=0.05,
            head_length=0.05,
            fc=colors["edge"],
            ec=colors["edge"],
            alpha=0.5,
        )

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors["api"], label="API Layer"),
        mpatches.Patch(color=colors["core"], label="Core Components"),
        mpatches.Patch(color=colors["repr"], label="Representations"),
        mpatches.Patch(color=colors["analysis"], label="Analysis Tools"),
        mpatches.Patch(color=colors["ext"], label="Extensions"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1))

    plt.tight_layout()
    return fig


def main():
    """Generate and save the architecture diagram."""
    print("🎨 Generating BooFun architecture diagram...")

    try:
        fig = create_architecture_diagram()

        # Save as PNG for README
        fig.savefig(
            "docs/architecture_diagram.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print("   ✅ PNG diagram saved to docs/architecture_diagram.png")

        # Save as SVG for documentation
        fig.savefig(
            "docs/architecture_diagram.svg",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print("   ✅ SVG diagram saved to docs/architecture_diagram.svg")

        plt.close()

        print("\n📖 Architecture diagram generated successfully!")
        print("   Use in README: ![Architecture](docs/architecture_diagram.png)")
        print("   Use in docs: .. image:: architecture_diagram.svg")

    except ImportError:
        print("   ⚠️  Matplotlib required for diagram generation")
        print("   Install with: pip install matplotlib")
    except Exception as e:
        print(f"   ❌ Error generating diagram: {e}")


if __name__ == "__main__":
    main()
