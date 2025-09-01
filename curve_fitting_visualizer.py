import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit
import csv
import webbrowser
from PIL import Image, ImageTk

class CurveFittingVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Curve Fitting Visualization Tool")
        self.root.geometry("1200x800")
        
        # Data storage
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.fitted_curves = {}
        
        # Create main frames
        self.control_frame = ttk.LabelFrame(root, text="Controls", padding=10)
        self.visualization_frame = ttk.Frame(root)
        self.education_frame = ttk.LabelFrame(root, text="Educational Content", padding=10)
        
        # Grid layout
        self.control_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.visualization_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        self.education_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        # Configure grid weights
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        
        # Initialize components
        self.setup_control_panel()
        self.setup_visualization()
        self.setup_education_panel()
        
        # Load initial sample data
        self.load_sample_data()

    def setup_control_panel(self):
        """Set up the control panel with all user inputs"""
        # Data input section
        input_frame = ttk.LabelFrame(self.control_frame, text="Data Input", padding=10)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(input_frame, text="Enter Data Manually", command=self.manual_data_entry).pack(fill=tk.X, pady=2)
        ttk.Button(input_frame, text="Import from CSV", command=self.import_csv).pack(fill=tk.X, pady=2)
        ttk.Button(input_frame, text="Generate Sample Data", command=self.load_sample_data).pack(fill=tk.X, pady=2)
        ttk.Button(input_frame, text="Clear Data", command=self.clear_data).pack(fill=tk.X, pady=2)
        
        # Model selection section
        model_frame = ttk.LabelFrame(self.control_frame, text="Fitting Models", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        self.model_var = tk.StringVar(value="Linear")
        models = [
            ("Linear", "linear"),
            ("Quadratic", "quadratic"),
            ("Cubic", "cubic"),
            ("Exponential", "exponential"),
            ("Power Law", "power"),
            ("Logarithmic", "logarithmic")
        ]
        
        for text, mode in models:
            ttk.Radiobutton(model_frame, text=text, variable=self.model_var, value=mode).pack(anchor=tk.W)
        
        # Parameter adjustment
        param_frame = ttk.LabelFrame(self.control_frame, text="Model Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=5)
        
        self.param_entries = []
        for i in range(3):  # Up to 3 parameters
            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"Param {i+1}:").pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=10)
            entry.pack(side=tk.RIGHT)
            self.param_entries.append(entry)
        
        ttk.Button(param_frame, text="Auto Fit", command=self.auto_fit).pack(fill=tk.X, pady=5)
        ttk.Button(param_frame, text="Manual Adjust", command=self.manual_adjust).pack(fill=tk.X, pady=2)
        
        # Analysis options
        analysis_frame = ttk.LabelFrame(self.control_frame, text="Analysis Tools", padding=10)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="Show Residuals", command=self.show_residuals).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Compare Models", command=self.compare_models).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Calculate Error Metrics", command=self.show_error_metrics).pack(fill=tk.X, pady=2)
        
        # Export options
        export_frame = ttk.LabelFrame(self.control_frame, text="Export Results", padding=10)
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Save Plot Image", command=self.export_plot).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Save Equation Data", command=self.export_equation).pack(fill=tk.X, pady=2)

    def setup_visualization(self):
        """Set up the visualization area with matplotlib canvas"""
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.visualization_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add click event for interactive data entry
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Initialize with empty plot
        self.update_plot()

    def setup_education_panel(self):
        """Set up the educational content panel"""
        # Create notebook for different educational tabs
        self.edu_notebook = ttk.Notebook(self.education_frame)
        self.edu_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Theory tab
        theory_frame = ttk.Frame(self.edu_notebook)
        self.theory_text = tk.Text(theory_frame, wrap=tk.WORD, height=10)
        scroll = ttk.Scrollbar(theory_frame, command=self.theory_text.yview)
        self.theory_text.configure(yscrollcommand=scroll.set)
        
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.theory_text.pack(fill=tk.BOTH, expand=True)
        
        # Tutorial tab
        tutorial_frame = ttk.Frame(self.edu_notebook)
        self.tutorial_text = tk.Text(tutorial_frame, wrap=tk.WORD, height=10)
        scroll_tut = ttk.Scrollbar(tutorial_frame, command=self.tutorial_text.yview)
        self.tutorial_text.configure(yscrollcommand=scroll_tut.set)
        
        scroll_tut.pack(side=tk.RIGHT, fill=tk.Y)
        self.tutorial_text.pack(fill=tk.BOTH, expand=True)
        
        # Equations tab
        equation_frame = ttk.Frame(self.edu_notebook)
        self.equation_text = tk.Text(equation_frame, wrap=tk.WORD, height=10)
        scroll_eq = ttk.Scrollbar(equation_frame, command=self.equation_text.yview)
        self.equation_text.configure(yscrollcommand=scroll_eq.set)
        
        scroll_eq.pack(side=tk.RIGHT, fill=tk.Y)
        self.equation_text.pack(fill=tk.BOTH, expand=True)
        
        # Add tabs to notebook
        self.edu_notebook.add(theory_frame, text="Theory")
        self.edu_notebook.add(tutorial_frame, text="Tutorial")
        self.edu_notebook.add(equation_frame, text="Equations")
        
        # Load educational content
        self.load_educational_content()

    def load_educational_content(self):
        """Load educational content into the info panels"""
        # Theory content
        theory_content = """CURVE FITTING THEORY

1. What is Curve Fitting?
Curve fitting is the process of constructing a curve (mathematical function) that has the best fit to a series of data points.

2. Purpose:
- Describe the relationship between variables
- Predict values between known data points (interpolation)
- Predict values beyond known data points (extrapolation)

3. Common Methods:
- Linear Regression: Fits a straight line (y = ax + b)
- Polynomial Regression: Fits polynomial functions
- Nonlinear Regression: Fits exponential, logarithmic, etc.
"""
        self.theory_text.insert(tk.END, theory_content)
        self.theory_text.config(state=tk.DISABLED)
        
        # Tutorial content
        tutorial_content = """STEP-BY-STEP TUTORIAL

1. Getting Started:
- Enter data manually or import from CSV
- Click on the plot to add/modify data points

2. Fitting a Curve:
- Select a model type from the controls
- Click 'Auto Fit' to automatically fit the curve
- Adjust parameters manually if needed

3. Analyzing Results:
- View residuals to assess fit quality
- Compare different models
- Export results for reports
"""
        self.tutorial_text.insert(tk.END, tutorial_content)
        self.tutorial_text.config(state=tk.DISABLED)
        
        # Equation content
        equation_content = """MODEL EQUATIONS

1. Linear: y = a*x + b
   - a: slope
   - b: y-intercept

2. Quadratic: y = a*x² + b*x + c
   - a, b, c: polynomial coefficients

3. Exponential: y = a*exp(b*x)
   - a: initial value
   - b: growth rate

4. Power Law: y = a*x^b
   - a: coefficient
   - b: exponent

5. Logarithmic: y = a*ln(x) + b
   - a: scale factor
   - b: offset
"""
        self.equation_text.insert(tk.END, equation_content)
        self.equation_text.config(state=tk.DISABLED)

    def load_sample_data(self):
        """Load sample data for demonstration"""
        self.x_data = np.linspace(0, 10, 20)
        self.y_data = 2.5 * self.x_data**2 - 3.2 * self.x_data + 1.5 + np.random.normal(0, 5, len(self.x_data))
        self.update_plot()

    def clear_data(self):
        """Clear all data points"""
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.fitted_curves = {}
        self.update_plot()

    def manual_data_entry(self):
        """Open dialog for manual data entry"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Data Points")
        
        tk.Label(dialog, text="Enter X and Y values (comma-separated):").pack(pady=5)
        
        x_frame = tk.Frame(dialog)
        tk.Label(x_frame, text="X values:").pack(side=tk.LEFT)
        x_entry = tk.Entry(x_frame, width=40)
        x_entry.pack(side=tk.RIGHT)
        x_frame.pack(pady=2)
        
        y_frame = tk.Frame(dialog)
        tk.Label(y_frame, text="Y values:").pack(side=tk.LEFT)
        y_entry = tk.Entry(y_frame, width=40)
        y_entry.pack(side=tk.RIGHT)
        y_frame.pack(pady=2)
        
        def apply_data():
            try:
                x_str = x_entry.get().strip()
                y_str = y_entry.get().strip()
                
                if x_str and y_str:
                    x = np.array([float(val.strip()) for val in x_str.split(',') if val.strip()])
                    y = np.array([float(val.strip()) for val in y_str.split(',') if val.strip()])
                    
                    if len(x) == len(y):
                        self.x_data = x
                        self.y_data = y
                        self.update_plot()
                        dialog.destroy()
                    else:
                        messagebox.showerror("Error", "X and Y must have the same number of values")
                else:
                    messagebox.showerror("Error", "Please enter values for both X and Y")
            except ValueError:
                messagebox.showerror("Error", "Invalid numeric values entered")
        
        tk.Button(dialog, text="Apply", command=apply_data).pack(pady=10)

    def import_csv(self):
        """Import data from CSV file"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    data = list(reader)
                    
                    # Try to automatically detect data format
                    if len(data[0]) >= 2:  # Assume first two columns are x and y
                        x = []
                        y = []
                        for row in data:
                            if len(row) >= 2:
                                try:
                                    x.append(float(row[0]))
                                    y.append(float(row[1]))
                                except ValueError:
                                    continue
                        
                        if x and y and len(x) == len(y):
                            self.x_data = np.array(x)
                            self.y_data = np.array(y)
                            self.update_plot()
                        else:
                            messagebox.showerror("Error", "Could not parse valid X,Y pairs from CSV")
                    else:
                        messagebox.showerror("Error", "CSV file must contain at least 2 columns of numeric data")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read CSV file: {str(e)}")

    def on_plot_click(self, event):
        """Handle mouse clicks on the plot for interactive data entry"""
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            
            # Check if we're near an existing point to modify it
            modify_index = None
            if len(self.x_data) > 0:
                distances = np.sqrt((self.x_data - x)**2 + (self.y_data - y)**2)
                min_dist = np.min(distances)
                if min_dist < 0.5:  # Threshold for point modification
                    modify_index = np.argmin(distances)
            
            if modify_index is not None:
                # Modify existing point
                self.x_data[modify_index] = x
                self.y_data[modify_index] = y
            else:
                # Add new point
                self.x_data = np.append(self.x_data, x)
                self.y_data = np.append(self.y_data, y)
            
            self.update_plot()

    def auto_fit(self):
        """Automatically fit the selected model to the data"""
        if len(self.x_data) == 0:
            messagebox.showerror("Error", "No data points to fit")
            return
        
        model_type = self.model_var.get()
        
        try:
            if model_type == "linear":
                popt, _ = curve_fit(self.linear_func, self.x_data, self.y_data)
                equation = f"y = {popt[0]:.4f}x + {popt[1]:.4f}"
                y_fit = self.linear_func(self.x_data, *popt)
            
            elif model_type == "quadratic":
                popt, _ = curve_fit(self.quadratic_func, self.x_data, self.y_data)
                equation = f"y = {popt[0]:.4f}x² + {popt[1]:.4f}x + {popt[2]:.4f}"
                y_fit = self.quadratic_func(self.x_data, *popt)
            
            elif model_type == "cubic":
                popt, _ = curve_fit(self.cubic_func, self.x_data, self.y_data)
                equation = f"y = {popt[0]:.4f}x³ + {popt[1]:.4f}x² + {popt[2]:.4f}x + {popt[3]:.4f}"
                y_fit = self.cubic_func(self.x_data, *popt)
            
            elif model_type == "exponential":
                # Initial guess for better convergence
                p0 = [1, 0.1]
                popt, _ = curve_fit(self.exponential_func, self.x_data, self.y_data, p0=p0, maxfev=5000)
                equation = f"y = {popt[0]:.4f}e^({popt[1]:.4f}x)"
                y_fit = self.exponential_func(self.x_data, *popt)
            
            elif model_type == "power":
                popt, _ = curve_fit(self.power_func, self.x_data, self.y_data)
                equation = f"y = {popt[0]:.4f}x^{popt[1]:.4f}"
                y_fit = self.power_func(self.x_data, *popt)
            
            elif model_type == "logarithmic":
                # Filter out x <= 0 for logarithmic
                valid = self.x_data > 0
                if np.sum(valid) < 2:
                    messagebox.showerror("Error", "Logarithmic fit requires positive x values")
                    return
                
                popt, _ = curve_fit(self.logarithmic_func, self.x_data[valid], self.y_data[valid])
                equation = f"y = {popt[0]:.4f}ln(x) + {popt[1]:.4f}"
                y_fit = np.zeros_like(self.x_data)
                y_fit[valid] = self.logarithmic_func(self.x_data[valid], *popt)
            
            self.fitted_curves[model_type] = {
                'equation': equation,
                'y_fit': y_fit,
                'params': popt
            }
            
            self.update_plot()
            
            # Update parameter entries
            for i, param in enumerate(popt):
                if i < len(self.param_entries):
                    self.param_entries[i].delete(0, tk.END)
                    self.param_entries[i].insert(0, f"{param:.4f}")
        
        except Exception as e:
            messagebox.showerror("Fit Error", f"Failed to fit {model_type} model: {str(e)}")

    def manual_adjust(self):
        """Manually adjust the parameters of the current model"""
        model_type = self.model_var.get()
        
        if model_type not in self.fitted_curves:
            messagebox.showerror("Error", "No fitted curve to adjust. Please run Auto Fit first.")
            return
        
        try:
            params = []
            for entry in self.param_entries:
                val = entry.get()
                if val:
                    params.append(float(val))
                else:
                    params.append(0.0)
            
            if model_type == "linear" and len(params) >= 2:
                y_fit = self.linear_func(self.x_data, *params[:2])
                equation = f"y = {params[0]:.4f}x + {params[1]:.4f}"
            
            elif model_type == "quadratic" and len(params) >= 3:
                y_fit = self.quadratic_func(self.x_data, *params[:3])
                equation = f"y = {params[0]:.4f}x² + {params[1]:.4f}x + {params[2]:.4f}"
            
            elif model_type == "cubic" and len(params) >= 4:
                y_fit = self.cubic_func(self.x_data, *params[:4])
                equation = f"y = {params[0]:.4f}x³ + {params[1]:.4f}x² + {params[2]:.4f}x + {params[3]:.4f}"
            
            elif model_type == "exponential" and len(params) >= 2:
                y_fit = self.exponential_func(self.x_data, *params[:2])
                equation = f"y = {params[0]:.4f}e^({params[1]:.4f}x)"
            
            elif model_type == "power" and len(params) >= 2:
                y_fit = self.power_func(self.x_data, *params[:2])
                equation = f"y = {params[0]:.4f}x^{params[1]:.4f}"
            
            elif model_type == "logarithmic" and len(params) >= 2:
                valid = self.x_data > 0
                y_fit = np.zeros_like(self.x_data)
                y_fit[valid] = self.logarithmic_func(self.x_data[valid], *params[:2])
                equation = f"y = {params[0]:.4f}ln(x) + {params[1]:.4f}"
            
            else:
                messagebox.showerror("Error", "Not enough parameters provided for this model")
                return
            
            self.fitted_curves[model_type] = {
                'equation': equation,
                'y_fit': y_fit,
                'params': params
            }
            
            self.update_plot()
        
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values. Please enter numbers only.")

    def show_residuals(self):
        """Display residuals plot"""
        model_type = self.model_var.get()
        
        if model_type not in self.fitted_curves:
            messagebox.showerror("Error", "No fitted curve to analyze. Please run Auto Fit first.")
            return
        
        residuals = self.y_data - self.fitted_curves[model_type]['y_fit']
        
        # Create a new figure for residuals
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Plot data and fit
        ax1.scatter(self.x_data, self.y_data, label='Data')
        ax1.plot(self.x_data, self.fitted_curves[model_type]['y_fit'], 
                'r-', label=f"Fit: {self.fitted_curves[model_type]['equation']}")
        ax1.set_title(f"{model_type.capitalize()} Fit")
        ax1.legend()
        ax1.grid(True)
        
        # Plot residuals
        ax2.scatter(self.x_data, residuals, color='green')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title("Residuals Plot")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Residuals")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def compare_models(self):
        """Compare different fitting models"""
        if len(self.x_data) == 0:
            messagebox.showerror("Error", "No data points to analyze")
            return
        
        models_to_compare = [
            ("Linear", self.linear_func, [0, 0]),
            ("Quadratic", self.quadratic_func, [0, 0, 0]),
            ("Exponential", self.exponential_func, [1, 0.1]),
            ("Power Law", self.power_func, [1, 1])
        ]
        
        results = []
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot original data
        ax.scatter(self.x_data, self.y_data, color='black', label='Data', zorder=5)
        
        # Fit and plot each model
        for name, func, p0 in models_to_compare:
            try:
                popt, _ = curve_fit(func, self.x_data, self.y_data, p0=p0, maxfev=5000)
                y_fit = func(self.x_data, *popt)
                
                # Calculate R-squared
                residuals = self.y_data - y_fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((self.y_data - np.mean(self.y_data))**2)
                r_squared = 1 - (ss_res / ss_tot)
                
                results.append((name, r_squared))
                ax.plot(self.x_data, y_fit, label=f"{name} (R²={r_squared:.3f})")
            
            except:
                continue
        
        ax.set_title("Model Comparison")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True)
        
        # Create results table
        results.sort(key=lambda x: x[1], reverse=True)  # Sort by R-squared
        
        table_text = "Model Comparison Results:\n\n"
        table_text += "{:<12} {:<10}\n".format("Model", "R-squared")
        table_text += "-"*22 + "\n"
        for name, r2 in results:
            table_text += "{:<12} {:.4f}\n".format(name, r2)
        
        # Show plot and results
        plt.show()
        messagebox.showinfo("Model Comparison", table_text)

    def show_error_metrics(self):
        """Calculate and display error metrics for the current fit"""
        model_type = self.model_var.get()
        
        if model_type not in self.fitted_curves:
            messagebox.showerror("Error", "No fitted curve to analyze. Please run Auto Fit first.")
            return
        
        y_fit = self.fitted_curves[model_type]['y_fit']
        residuals = self.y_data - y_fit
        
        # Calculate metrics
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.y_data - np.mean(self.y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Create report
        report = f"""Error Metrics for {model_type.capitalize()} Fit:
        
Equation: {self.fitted_curves[model_type]['equation']}

Mean Squared Error (MSE): {mse:.4f}
Root Mean Squared Error (RMSE): {rmse:.4f}
Mean Absolute Error (MAE): {mae:.4f}
R-squared: {r_squared:.4f}
"""
        messagebox.showinfo("Error Metrics", report)

    def export_plot(self):
        """Export the current plot to an image file"""
        if len(self.x_data) == 0:
            messagebox.showerror("Error", "No data to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("PDF Document", "*.pdf")],
            title="Save Plot As")
        
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved successfully to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

    def export_equation(self):
        """Export the equation and parameters to a text file"""
        model_type = self.model_var.get()
        
        if model_type not in self.fitted_curves:
            messagebox.showerror("Error", "No fitted curve to export. Please run Auto Fit first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("CSV File", "*.csv")],
            title="Save Equation As")
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(f"Curve Fitting Results\n\n")
                    f.write(f"Model Type: {model_type.capitalize()}\n")
                    f.write(f"Equation: {self.fitted_curves[model_type]['equation']}\n\n")
                    
                    if file_path.endswith('.csv'):
                        # Write CSV format
                        f.write("Parameter,Value\n")
                        for i, param in enumerate(self.fitted_curves[model_type]['params']):
                            f.write(f"Param {i+1},{param:.6f}\n")
                    else:
                        # Write text format
                        f.write("Parameters:\n")
                        for i, param in enumerate(self.fitted_curves[model_type]['params']):
                            f.write(f"  Param {i+1}: {param:.6f}\n")
                
                messagebox.showinfo("Success", f"Equation data saved successfully to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save equation data: {str(e)}")

    def update_plot(self):
        """Update the plot with current data and fits"""
        self.ax.clear()
        
        # Plot data points if they exist
        if len(self.x_data) > 0:
            self.ax.scatter(self.x_data, self.y_data, label='Data Points', color='blue', zorder=5)
        
        # Plot fitted curves
        for model_type, fit_data in self.fitted_curves.items():
            if len(self.x_data) == len(fit_data['y_fit']):
                self.ax.plot(self.x_data, fit_data['y_fit'], 
                            label=f"{model_type.capitalize()}: {fit_data['equation']}")
        
        # Set plot elements
        self.ax.set_title("Curve Fitting Visualization")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        
        if len(self.x_data) > 0 or self.fitted_curves:
            self.ax.legend()
            self.ax.grid(True)
        
        self.canvas.draw()

    # Mathematical model functions
    @staticmethod
    def linear_func(x, a, b):
        return a * x + b
    
    @staticmethod
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c
    
    @staticmethod
    def cubic_func(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d
    
    @staticmethod
    def exponential_func(x, a, b):
        return a * np.exp(b * x)
    
    @staticmethod
    def power_func(x, a, b):
        return a * np.power(x, b)
    
    @staticmethod
    def logarithmic_func(x, a, b):
        return a * np.log(x) + b

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CurveFittingVisualizer(root)
    root.mainloop()