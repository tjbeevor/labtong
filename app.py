import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

class LPBFPredictor:
    def __init__(self, data_file):
        """Initialize the predictor with data file"""
        self.df = pd.read_csv(data_file, header=1)
        self.clean_data()
        self.train_model()
        
    def clean_data(self):
        """Clean and prepare the data with more inclusive handling"""
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Function to safely convert values
        def safe_numeric_convert(value):
            if pd.isna(value):
                return np.nan
            try:
                # Handle string values
                if isinstance(value, str):
                    # Remove common units and special characters
                    value = value.replace('Mpa', '').replace('HV', '').strip()
                    # Handle division by zero markers
                    if value in ['#DIV/0!', '#NUM!', 'inf', '-inf']:
                        return np.nan
                return float(value)
            except:
                return np.nan
        
        # Show initial data info
        st.write(f"Initial data points: {len(self.df)}")
        
        # Convert columns
        numeric_columns = ['YS', 'UTS', 'Elongation', 'power', 'speed', 'Hatch', 'thickness']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(safe_numeric_convert)
        
        # Create analysis dataframe
        self.analysis_df = self.df.copy()
        
        # Calculate additional properties
        self.calculate_material_properties()
        
        # Set process windows based on valid data
        valid_mask = (self.analysis_df['power'].notna() & 
                     self.analysis_df['speed'].notna())
        
        self.process_windows = {
            'power': {
                'min': self.analysis_df[valid_mask]['power'].quantile(0.25),
                'max': self.analysis_df[valid_mask]['power'].quantile(0.75),
                'optimal': self.analysis_df[valid_mask]['power'].median()
            },
            'speed': {
                'min': self.analysis_df[valid_mask]['speed'].quantile(0.25),
                'max': self.analysis_df[valid_mask]['speed'].quantile(0.75),
                'optimal': self.analysis_df[valid_mask]['speed'].median()
            },
            'p/v': {
                'min': (self.analysis_df[valid_mask]['power'] / 
                       self.analysis_df[valid_mask]['speed']).quantile(0.25),
                'max': (self.analysis_df[valid_mask]['power'] / 
                       self.analysis_df[valid_mask]['speed']).quantile(0.75),
                'optimal': (self.analysis_df[valid_mask]['power'] / 
                          self.analysis_df[valid_mask]['speed']).median()
            }
        }
        
        st.write(f"Data points available for analysis: {sum(valid_mask)}")

    def calculate_material_properties(self):
        """Calculate additional material properties"""
        # Calculate P/V ratio
        self.analysis_df['p/v'] = self.analysis_df['power'] / self.analysis_df['speed']
        
        # Calculate YS/UTS ratio where possible
        if all(col in self.analysis_df.columns for col in ['YS', 'UTS']):
            self.analysis_df['YS_UTS_ratio'] = self.analysis_df['YS'] / self.analysis_df['UTS']
        
        # Calculate volumetric energy density where possible
        if all(col in self.analysis_df.columns for col in ['power', 'speed', 'Hatch', 'thickness']):
            self.analysis_df['Energy_Density'] = (
                self.analysis_df['power'] / 
                (self.analysis_df['speed'] * self.analysis_df['Hatch'] * self.analysis_df['thickness'])
            )
        
        # Calculate quality index where possible
        if all(col in self.analysis_df.columns for col in ['UTS', 'Elongation']):
            self.analysis_df['quality_index'] = (
                self.analysis_df['UTS'] * np.log(self.analysis_df['Elongation'])
            )

    def train_model(self):
        """Train the prediction model"""
        # Prepare features
        features = ['power', 'speed', 'p/v']
        if 'Energy_Density' in self.analysis_df.columns:
            features.append('Energy_Density')
            
        valid_mask = self.analysis_df[features + ['YS']].notna().all(axis=1)
        
        X = self.analysis_df[valid_mask][features]
        y = self.analysis_df[valid_mask]['YS']
        
        if len(X) < 10:
            st.error("Insufficient data for model training")
            return
        
        # Split and train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate metrics
        self.model_metrics = {
            'train_score': r2_score(self.y_train, self.model.predict(self.X_train)),
            'test_score': r2_score(self.y_test, self.model.predict(self.X_test)),
            'mae': mean_absolute_error(self.y_test, self.model.predict(self.X_test))
        }
    
    def predict_strength(self, power, speed):
        """Predict yield strength for given parameters"""
        p_v = power/speed
        features = pd.DataFrame([[power, speed, p_v]], 
                              columns=['power', 'speed', 'p/v'])
        if 'Energy_Density' in self.analysis_df.columns:
            # Use median values for missing parameters
            hatch = self.analysis_df['Hatch'].median()
            thickness = self.analysis_df['thickness'].median()
            energy_density = power / (speed * hatch * thickness)
            features['Energy_Density'] = energy_density
            
        return self.model.predict(features)[0]
    
    def analyze_microstructure(self, power, speed, energy_density):
        """Predict microstructural features based on parameters"""
        microstructure_analysis = {}
        
        # Cooling rate estimation (simplified)
        cooling_rate = speed * 1000 / power  # approximate relationship
        
        if cooling_rate > 10**6:
            microstructure_analysis['primary_feature'] = "Fine cellular structure"
            microstructure_analysis['SDAS'] = "< 1 μm"
        elif cooling_rate > 10**5:
            microstructure_analysis['primary_feature'] = "Cellular-dendritic"
            microstructure_analysis['SDAS'] = "1-5 μm"
        else:
            microstructure_analysis['primary_feature'] = "Coarse dendritic"
            microstructure_analysis['SDAS'] = "> 5 μm"
        
        # Melt pool analysis
        if energy_density > 100:
            microstructure_analysis['melt_pool'] = "Risk of keyholing"
        elif energy_density < 40:
            microstructure_analysis['melt_pool'] = "Risk of lack of fusion"
        else:
            microstructure_analysis['melt_pool'] = "Conduction mode melting"
        
        return microstructure_analysis
    
    def predict_defects(self, power, speed):
        """Predict likely defect types based on parameters"""
        p_v_ratio = power/speed
        defects = []
        
        # Low energy defects
        if p_v_ratio < self.process_windows['p/v']['min']:
            defects.extend([
                "Lack of fusion porosity",
                "Incomplete melting",
                "Poor layer adhesion"
            ])
        
        # High energy defects
        elif p_v_ratio > self.process_windows['p/v']['max']:
            defects.extend([
                "Keyhole porosity",
                "Residual stress accumulation",
                "Surface roughness issues"
            ])
        
        # Speed-related defects
        if speed > self.process_windows['speed']['max']:
            defects.extend([
                "Balling phenomenon",
                "Discontinuous tracks"
            ])
        elif speed < self.process_windows['speed']['min']:
            defects.extend([
                "Excessive heat accumulation",
                "Thermal distortion"
            ])
        
        return defects
    
    def analyze_parameters(self, power, speed):
        """Analyze the given parameters"""
        p_v_ratio = power/speed
        
        issues = []
        process_window = []
        problems = []
        
        # Analysis logic
        if power < self.process_windows['power']['min']:
            issues.append("Power is below optimal range")
            problems.append("Insufficient energy for complete melting")
        elif power > self.process_windows['power']['max']:
            issues.append("Power is above optimal range")
            problems.append("Risk of keyholing defects")
        
        if speed < self.process_windows['speed']['min']:
            issues.append("Speed is below optimal range")
            problems.append("Excessive heat accumulation")
        elif speed > self.process_windows['speed']['max']:
            issues.append("Speed is above optimal range")
            problems.append("Insufficient melting likely")
        
        # Process window information
        process_window.extend([
            f"Optimal power range: {self.process_windows['power']['min']:.0f}-{self.process_windows['power']['max']:.0f}W",
            f"Optimal speed range: {self.process_windows['speed']['min']:.0f}-{self.process_windows['speed']['max']:.0f}mm/s",
            f"Current P/V ratio: {p_v_ratio:.2f} J/mm"
        ])
        
        return {
            'issues': issues,
            'process_window': process_window,
            'problems': problems
        }
    
    def create_process_window_plot(self, current_power, current_speed):
        """Create process window visualization"""
        fig = go.Figure()
        
        # Plot all valid data points
        valid_mask = (self.df['power'].notna() & 
                     self.df['speed'].notna() & 
                     self.df['YS'].notna())
        
        fig.add_trace(go.Scatter(
            x=self.df[valid_mask]['power'],
            y=self.df[valid_mask]['speed'],
            mode='markers',
            name='Dataset Points',
            marker=dict(
                size=8,
                color=self.df[valid_mask]['YS'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Yield Strength (MPa)')
            ),
            hovertemplate='Power: %{x}W<br>Speed: %{y}mm/s<br>YS: %{marker.color:.0f}MPa<extra></extra>'
        ))
        
        # Add current point
        fig.add_trace(go.Scatter(
            x=[current_power],
            y=[current_speed],
            mode='markers',
            name='Selected Parameters',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            )
        ))
        
        # Add process window
        fig.add_shape(
            type="rect",
            x0=self.process_windows['power']['min'],
            y0=self.process_windows['speed']['min'],
            x1=self.process_windows['power']['max'],
            y1=self.process_windows['speed']['max'],
            line=dict(color="rgba(0,255,0,0.5)"),
            fillcolor="rgba(0,255,0,0.1)"
        )
        
        fig.update_layout(
            title='Process Window Analysis',
            xaxis_title='Power (W)',
            yaxis_title='Scan Speed (mm/s)',
            hovermode='closest'
        )
        
        return fig

def main():
    st.set_page_config(page_title="LPBF Parameter Predictor", layout="wide")
    
    st.title("LPBF AlSi10Mg Parameter Predictor")
    st.write("""
    This tool predicts mechanical properties and analyzes process parameters for LPBF AlSi10Mg. 
    Upload your dataset to begin analysis.
    """)
    
    uploaded_file = st.file_uploader("Upload LPBF data CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            predictor = LPBFPredictor(uploaded_file)
            
            st.sidebar.header("Parameter Input")
            power = st.sidebar.slider("Laser Power (W)", 100, 800, 350)
            speed = st.sidebar.slider("Scan Speed (mm/s)", 100, 2000, 1000)
            
            # Calculate energy density for microstructure prediction
            if 'Hatch' in predictor.analysis_df.columns and 'thickness' in predictor.analysis_df.columns:
                hatch = predictor.analysis_df['Hatch'].median()
                thickness = predictor.analysis_df['thickness'].median()
                energy_density = power / (speed * hatch * thickness)
            else:
                energy_density = power / speed  # simplified calculation
            
            predicted_ys = predictor.predict_strength(power, speed)
            analysis = predictor.analyze_parameters(power, speed)
            microstructure = predictor.analyze_microstructure(power, speed, energy_density)
            defects = predictor.predict_defects(power, speed)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Prediction Results")
                st.metric("Predicted Yield Strength", f"{predicted_ys:.1f} MPa")
                
                st.subheader("Process Window Analysis")
                for info in analysis['process_window']:
                    st.info(info)
                
                st.subheader("Predicted Microstructure")
                for feature, value in microstructure.items():
                    st.write(f"**{feature.replace('_', ' ').title()}:** {value}")
            
            with col2:
                st.subheader("Potential Issues")
                if analysis['issues']:
                    for issue in analysis['issues']:
                        st.warning(issue)
                else:
                    st.success("Parameters are within typical ranges")
                
                st.subheader("Predicted Defects")
                if defects:
                    for defect in defects:
                        st.error(defect)
                else:
                    st.success("No significant defects predicted")
                
                st.subheader("Model Performance")
                st.write(f"Training R² Score: {predictor.model_metrics['train_score']:.3f}")
                st.write(f"Testing R² Score: {predictor.model_metrics['test_score']:.3f}")
                st.write(f"Mean Absolute Error: {predictor.model_metrics['mae']:.1f} MPa")
            
            # Process Window Plot
            st.header("Process Window Visualization")
            st.plotly_chart(predictor.create_process_window_plot(power, speed), use_container_width=True)
            
            # Additional Analysis Section
            st.header("Additional Analysis")
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Property Relationships")
                if all(col in predictor.analysis_df.columns for col in ['UTS', 'Elongation']):
                    fig = px.scatter(
                        predictor.analysis_df,
                        x='UTS',
                        y='Elongation',
                        color='Energy_Density' if 'Energy_Density' in predictor.analysis_df.columns else None,
                        title="Strength-Ductility Relationship",
                        labels={
                            'UTS': 'Ultimate Tensile Strength (MPa)',
                            'Elongation': 'Elongation (%)',
                            'Energy_Density': 'Energy Density (J/mm³)'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                st.subheader("Quality Metrics")
                if 'quality_index' in predictor.analysis_df.columns:
                    fig = px.histogram(
                        predictor.analysis_df,
                        x='quality_index',
                        title="Quality Index Distribution",
                        nbins=30
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Material Properties Section
            st.header("Material Properties Analysis")
            col5, col6 = st.columns(2)
            
            with col5:
                if 'YS_UTS_ratio' in predictor.analysis_df.columns:
                    st.subheader("Strength Ratio Analysis")
                    fig = px.histogram(
                        predictor.analysis_df,
                        x='YS_UTS_ratio',
                        title="YS/UTS Ratio Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    median_ratio = predictor.analysis_df['YS_UTS_ratio'].median()
                    st.write(f"Median YS/UTS Ratio: {median_ratio:.2f}")
                    
                    if median_ratio < 0.7:
                        st.write("💡 Low YS/UTS ratio indicates good strain hardening capacity")
                    elif median_ratio > 0.9:
                        st.write("💡 High YS/UTS ratio indicates limited strain hardening")
            
            with col6:
                if 'Energy_Density' in predictor.analysis_df.columns:
                    st.subheader("Energy Density Analysis")
                    fig = px.scatter(
                        predictor.analysis_df,
                        x='Energy_Density',
                        y='YS',
                        title="Energy Density vs Yield Strength",
                        labels={
                            'Energy_Density': 'Energy Density (J/mm³)',
                            'YS': 'Yield Strength (MPa)'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download section
            st.header("Export Results")
            if st.button("Generate Report"):
                # Create report dataframe
                report_data = {
                    'Parameter': [
                        'Power (W)',
                        'Speed (mm/s)',
                        'P/V Ratio (J/mm)',
                        'Predicted YS (MPa)',
                        'Microstructure Type',
                        'Melt Pool Condition'
                    ],
                    'Value': [
                        power,
                        speed,
                        power/speed,
                        predicted_ys,
                        microstructure['primary_feature'],
                        microstructure['melt_pool']
                    ]
                }
                
                report_df = pd.DataFrame(report_data)
                csv = report_df.to_csv(index=False)
                
                st.download_button(
                    "Download Analysis Report",
                    csv,
                    "lpbf_analysis_report.csv",
                    "text/csv",
                    key='download-csv'
                )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Debug Information:")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, header=1)
                st.write("Columns in uploaded file:", df.columns.tolist())
                st.write("First few rows of data:")
                st.write(df.head())

if __name__ == "__main__":
    main()
