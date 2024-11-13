import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="LPBF AlSi10Mg Analysis", layout="wide")

def calculate_energy_density(power, speed, hatch, thickness):
    """Calculate volumetric energy density in J/mm³"""
    return power / (speed * hatch * thickness)

def process_data(df):
    """Process LPBF data with materials science focus"""
    processed_df = df.copy()
    
    # Convert numeric columns
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    
    # Calculate energy density where possible
    if all(col in processed_df.columns for col in ['power', 'speed', 'Hatch', 'thickness']):
        processed_df['Energy_Density'] = calculate_energy_density(
            processed_df['power'],
            processed_df['speed'],
            processed_df['Hatch'],
            processed_df['thickness']
        )
    
    # Calculate strength to weight ratio if possible
    if 'UTS' in processed_df.columns:
        processed_df['Specific_Strength'] = processed_df['UTS'] / 2.7  # AlSi10Mg density ≈ 2.7 g/cm³
    
    return processed_df

def main():
    st.title("LPBF AlSi10Mg Process-Property Analyzer")
    st.write("""
    Analyze the relationships between processing parameters, heat treatment, and mechanical properties 
    of LPBF AlSi10Mg. Upload your data to begin analysis.
    """)

    uploaded_file = st.file_uploader("Upload LPBF data (CSV)", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        processed_data = process_data(data)

        # Create main analysis sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "Process-Property Relationships",
            "Heat Treatment Analysis",
            "Property Optimization",
            "Quality Assessment"
        ])

        with tab1:
            st.header("Process-Property Relationships")
            
            # Energy Density Analysis
            st.subheader("Energy Density Impact")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Energy_Density' in processed_data.columns:
                    fig = px.scatter(
                        processed_data,
                        x='Energy_Density',
                        y='UTS',
                        color='Direction' if 'Direction' in processed_data.columns else None,
                        title="Energy Density vs Ultimate Tensile Strength",
                        labels={
                            'Energy_Density': 'Energy Density (J/mm³)',
                            'UTS': 'Ultimate Tensile Strength (MPa)'
                        }
                    )
                    st.plotly_chart(fig)
                    
                    # Add process window identification
                    optimal_energy = processed_data.loc[processed_data['UTS'].idxmax(), 'Energy_Density']
                    st.write(f"Optimal Energy Density: {optimal_energy:.2f} J/mm³")

            with col2:
                if 'Energy_Density' in processed_data.columns:
                    fig = px.scatter(
                        processed_data,
                        x='Energy_Density',
                        y='Elongation',
                        color='Direction' if 'Direction' in processed_data.columns else None,
                        title="Energy Density vs Elongation",
                        labels={
                            'Energy_Density': 'Energy Density (J/mm³)',
                            'Elongation': 'Elongation (%)'
                        }
                    )
                    st.plotly_chart(fig)

            # Process Parameter Map
            st.subheader("Process Parameter Map")
            if all(col in processed_data.columns for col in ['power', 'speed', 'UTS']):
                fig = px.scatter(
                    processed_data,
                    x='speed',
                    y='power',
                    color='UTS',
                    size='UTS',
                    title="Process Parameter Map",
                    labels={
                        'speed': 'Scan Speed (mm/s)',
                        'power': 'Laser Power (W)',
                        'UTS': 'Ultimate Tensile Strength (MPa)'
                    }
                )
                st.plotly_chart(fig)

        with tab2:
            st.header("Heat Treatment Analysis")
            
            # Heat Treatment Effect Analysis
            if 'solution temp' in processed_data.columns:
                st.subheader("Solution Treatment Effects")
                
                fig = px.box(
                    processed_data,
                    x=pd.qcut(processed_data['solution temp'], q=4),
                    y='UTS',
                    title="Solution Temperature Effect on Strength",
                    labels={
                        'solution temp': 'Solution Temperature (°C)',
                        'UTS': 'Ultimate Tensile Strength (MPa)'
                    }
                )
                st.plotly_chart(fig)

                # Heat Treatment Optimization Guide
                st.subheader("Heat Treatment Optimization")
                optimal_solution_temp = processed_data.loc[processed_data['UTS'].idxmax(), 'solution temp']
                st.write(f"Optimal Solution Temperature: {optimal_solution_temp:.0f}°C")

        with tab3:
            st.header("Property Optimization")
            
            # Property Distribution Analysis
            st.subheader("Strength-Ductility Balance")
            if all(col in processed_data.columns for col in ['UTS', 'Elongation']):
                fig = px.scatter(
                    processed_data,
                    x='UTS',
                    y='Elongation',
                    color='Energy_Density' if 'Energy_Density' in processed_data.columns else None,
                    title="Strength-Ductility Relationship",
                    labels={
                        'UTS': 'Ultimate Tensile Strength (MPa)',
                        'Elongation': 'Elongation (%)',
                        'Energy_Density': 'Energy Density (J/mm³)'
                    }
                )
                st.plotly_chart(fig)

            # Optimization Guidelines
            st.subheader("Process Optimization Guidelines")
            if 'UTS' in processed_data.columns:
                top_performers = processed_data.nlargest(5, 'UTS')
                st.write("Top 5 Performing Parameters:")
                st.dataframe(top_performers[['power', 'speed', 'UTS', 'YS', 'Elongation']].style.highlight_max())

        with tab4:
            st.header("Quality Assessment")
            
            # Quality Metrics
            st.subheader("Quality Metrics")
            if 'Density' in processed_data.columns:
                fig = px.histogram(
                    processed_data,
                    x='Density',
                    title="Part Density Distribution",
                    labels={'Density': 'Relative Density (%)'}
                )
                st.plotly_chart(fig)

            # Property Achievement Rate
            if 'UTS' in processed_data.columns:
                target_uts = 400  # Target UTS for AlSi10Mg
                achievement_rate = (processed_data['UTS'] >= target_uts).mean() * 100
                st.metric("Property Achievement Rate", f"{achievement_rate:.1f}%")

if __name__ == "__main__":
    main()
