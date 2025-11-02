import React, { useState } from "react";
import "./App.css";
import PrismaticBurst from "./js/prismatic-burst.jsx";
import Orb from "./js/orb.jsx";
import DarkVeil from "./js/DarkVeil.jsx";
import SplitText from "./js/SplitText.jsx";

function App() {
  const [veilVisible, setVeilVisible] = useState(true);
  const [orbExpanded, setOrbExpanded] = useState(false);
  const [orbVisible, setOrbVisible] = useState(true);
  const [showText, setShowText] = useState(false);

  // Function to start the transition sequence
  const startReveal = () => {
    // Fade veil out
    setVeilVisible(false);

    // After 5 seconds, fade veil back in and expand/fade orb
    setTimeout(() => {
      setVeilVisible(true);
      setOrbExpanded(true);

      // After orb finishes expanding, hide orb and show text
      setTimeout(() => {
        setOrbVisible(false);
        setShowText(true);
      }, 3000); // orb transition duration
    }, 5000);
  };

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        backgroundColor: "black",
      }}
    >
      {/* PrismaticBurst Background */}
      <PrismaticBurst
        animationType="rotate3d"
        intensity={1.5}
        speed={0.5}
        distort={3.6}
        paused={false}
        offset={{ x: 0, y: 0 }}
        hoverDampness={0.25}
        rayCount={30}
        mixBlendMode="lighten"
        colors={["#f0c8ff", "#773bcd", "#773bcd"]}
        style={{
          position: "absolute",
          inset: 0,
          opacity: veilVisible ? 0 : 1,
          transition: "opacity 1.5s ease-in-out",
        }}
      />

      {/* Dark Veil */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          opacity: veilVisible ? 1 : 0,
          transition: "opacity 1.5s ease-in-out",
          pointerEvents: veilVisible ? "auto" : "none",
          zIndex: 10,
        }}
      >
        <DarkVeil />
      </div>

      {/* Centered Orb Overlay */}
      {orbVisible && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: `translate(-50%, -50%) scale(${orbExpanded ? 5 : 1})`,
            transition: "transform 3s ease-in-out, opacity 3s ease-in-out",
            width: "600px",
            height: "600px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 20,
            opacity: orbExpanded ? 0 : 1,
          }}
        >
          <Orb hoverIntensity={0.5} rotateOnHover={true} hue={0} forceHoverState={false} />
        </div>
      )}

      {/* Main Text + Subtitle */}
      {showText && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            zIndex: 25,
            textAlign: "center",
          }}
        >
          {/* Main Text */}
          <SplitText
            text="Song Name - Artist Name"
            delay={70}
            duration={2}
            ease="elastic.out(1, 0.3)"
            splitType="chars"
            from={{ opacity: 0, y: 40 }}
            to={{ opacity: 1, y: 0 }}
            threshold={0.1}
            rootMargin="-100px"
            textAlign="center"
            className="split-text"
            style={{
              fontSize: "4rem",
              fontWeight: "bold",
              color: "white",
              fontFamily: "Helvetica, Arial, sans-serif",
            }}
          />

          {/* Subtitle / additional text */}
          <div
            style={{
              marginTop: "0.2rem",
              fontSize: "2rem",
              fontWeight: "300",
              fontFamily: "Helvetica, Arial, sans-serif",
              color: "white",
              opacity: 0,
              animation: "fadeIn 2s ease forwards",
              animationDelay: "2.5s", // starts after main text animation
            }}
          >
            Additional info or subtitle goes here
          </div>
        </div>
      )}

      {/* Example button to trigger the transition */}
      <button
        onClick={startReveal}
        style={{
          position: "absolute",
          bottom: "2rem",
          left: "50%",
          transform: "translateX(-50%)",
          padding: "1rem 2rem",
          fontSize: "1rem",
          borderRadius: "0.5rem",
          cursor: "pointer",
          zIndex: 30,
        }}
      >
        Start Reveal
      </button>
    </div>
  );
}

export default App;

