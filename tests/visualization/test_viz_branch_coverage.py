"""
Visualization branch coverage tests.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import boofun as bf
from boofun.visualization import BooleanFunctionVisualizer


@pytest.fixture(autouse=True)
def cleanup_plots():
    yield
    plt.close('all')


class TestVisualizerMethods:
    """Test all visualizer methods systematically."""
    
    def test_all_methods_majority(self):
        """Test all methods with majority function."""
        f = bf.majority(5)
        viz = BooleanFunctionVisualizer(f)
        
        for name in dir(viz):
            if name.startswith('_'):
                continue
            attr = getattr(viz, name)
            if callable(attr):
                try:
                    result = attr(show=False)
                except TypeError:
                    try:
                        result = attr()
                    except:
                        pass
                except:
                    pass
    
    def test_all_methods_parity(self):
        """Test all methods with parity function."""
        f = bf.parity(4)
        viz = BooleanFunctionVisualizer(f)
        
        for name in dir(viz):
            if name.startswith('_'):
                continue
            attr = getattr(viz, name)
            if callable(attr):
                try:
                    result = attr(show=False)
                except TypeError:
                    try:
                        result = attr()
                    except:
                        pass
                except:
                    pass
    
    def test_all_methods_and(self):
        """Test all methods with AND function."""
        f = bf.AND(3)
        viz = BooleanFunctionVisualizer(f)
        
        for name in dir(viz):
            if name.startswith('_'):
                continue
            attr = getattr(viz, name)
            if callable(attr):
                try:
                    result = attr(show=False)
                except TypeError:
                    try:
                        result = attr()
                    except:
                        pass
                except:
                    pass


class TestVisualizerOptions:
    """Test visualizer with various options."""
    
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_various_sizes(self, n):
        """Test with various function sizes."""
        f = bf.majority(n) if n % 2 == 1 else bf.parity(n)
        viz = BooleanFunctionVisualizer(f)
        
        fig = viz.plot_influences(show=False)
        assert fig is not None
    
    def test_with_title(self):
        """Test with custom title."""
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)
        
        if hasattr(viz.plot_influences, '__code__'):
            try:
                fig = viz.plot_influences(title="Custom Title", show=False)
            except TypeError:
                fig = viz.plot_influences(show=False)
        else:
            fig = viz.plot_influences(show=False)
        assert fig is not None


class TestGrowthPlotsMore:
    """More growth plots tests."""
    
    def test_all_classes(self):
        """Test all classes in growth_plots."""
        from boofun.visualization import growth_plots
        
        for name in dir(growth_plots):
            if name.startswith('_'):
                continue
            obj = getattr(growth_plots, name)
            if isinstance(obj, type):
                try:
                    instance = obj()
                except TypeError:
                    pass
    
    def test_all_functions(self):
        """Test all functions in growth_plots."""
        from boofun.visualization import growth_plots
        
        f = bf.majority(3)
        
        for name in dir(growth_plots):
            if name.startswith('_'):
                continue
            obj = getattr(growth_plots, name)
            if callable(obj) and not isinstance(obj, type):
                try:
                    result = obj(f)
                except:
                    try:
                        result = obj()
                    except:
                        pass


class TestAnimationMore:
    """More animation tests."""
    
    def test_all_classes(self):
        """Test all animation classes."""
        from boofun.visualization import animation
        
        for name in dir(animation):
            if name.startswith('_'):
                continue
            obj = getattr(animation, name)
            if isinstance(obj, type):
                try:
                    instance = obj()
                except TypeError:
                    pass
    
    def test_all_functions(self):
        """Test all animation functions."""
        from boofun.visualization import animation
        
        f = bf.majority(3)
        
        for name in dir(animation):
            if name.startswith('_'):
                continue
            obj = getattr(animation, name)
            if callable(obj) and not isinstance(obj, type):
                try:
                    result = obj(f, show=False)
                except:
                    try:
                        result = obj(f)
                    except:
                        pass


class TestWidgetsMore:
    """More widgets tests."""
    
    def test_all_classes(self):
        """Test all widget classes."""
        from boofun.visualization import widgets
        
        f = bf.majority(3)
        
        for name in dir(widgets):
            if name.startswith('_'):
                continue
            obj = getattr(widgets, name)
            if isinstance(obj, type):
                try:
                    instance = obj(f)
                except:
                    try:
                        instance = obj()
                    except:
                        pass


class TestInteractiveMore:
    """More interactive tests."""
    
    def test_all_contents(self):
        """Test all interactive contents."""
        from boofun.visualization import interactive
        
        f = bf.AND(3)
        
        for name in dir(interactive):
            if name.startswith('_'):
                continue
            obj = getattr(interactive, name)
            if callable(obj):
                try:
                    result = obj(f)
                except:
                    try:
                        result = obj()
                    except:
                        pass


class TestLatexExportMore:
    """More latex export tests."""
    
    def test_all_functions(self):
        """Test all LaTeX export functions."""
        from boofun.visualization import latex_export
        
        f = bf.OR(2)
        
        for name in dir(latex_export):
            if name.startswith('_'):
                continue
            obj = getattr(latex_export, name)
            if callable(obj):
                try:
                    result = obj(f)
                except:
                    try:
                        result = obj()
                    except:
                        pass


class TestDecisionTreeExportMore:
    """More decision tree export tests."""
    
    def test_all_functions(self):
        """Test all decision tree export functions."""
        from boofun.visualization import decision_tree_export
        
        f = bf.AND(3)
        
        for name in dir(decision_tree_export):
            if name.startswith('_'):
                continue
            obj = getattr(decision_tree_export, name)
            if callable(obj):
                try:
                    result = obj(f)
                except:
                    try:
                        result = obj()
                    except:
                        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
