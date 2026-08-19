export const GARMENT_CATALOG = [
  {
    id: 'silk-slip-dress',
    name: 'Cyber Silk Slip Midi Dress',
    category: 'Evening Wear',
    fabric: '100% Heavy Mulberry Silk',
    elasticity: 'Low (2% Lycra)',
    description: 'Sleek bias-cut midi dress with delicate spaghetti straps and subtle cowl neckline. Drapes fluidly along hips.',
    baseColor: '#E2E8F0',
    metallic: 0.1,
    roughness: 0.25,
    sizes: {
      XS: { chest: 81, waist: 63, hip: 88, length: 110 },
      S:  { chest: 85, waist: 67, hip: 92, length: 112 },
      M:  { chest: 89, waist: 71, hip: 96, length: 114 },
      L:  { chest: 94, waist: 76, hip: 101, length: 116 },
      XL: { chest: 100, waist: 82, hip: 107, length: 118 }
    },
    fitProfile: 'Regular Bias Fit',
    drapeFactor: 1.15
  },
  {
    id: 'sculpt-bodycon',
    name: 'Neo-Sculpt Ribbed Bodycon',
    category: 'Cocktail & Party',
    fabric: 'Compression Knit Blend',
    elasticity: 'High Stretch (18% Elastane)',
    description: 'Form-fitting architectural dress designed to contour the silhouette with supportive ribbed side-panels.',
    baseColor: '#00F0FF',
    metallic: 0.05,
    roughness: 0.4,
    sizes: {
      XS: { chest: 78, waist: 60, hip: 85, length: 98 },
      S:  { chest: 82, waist: 64, hip: 89, length: 100 },
      M:  { chest: 86, waist: 68, hip: 93, length: 102 },
      L:  { chest: 91, waist: 73, hip: 98, length: 104 },
      XL: { chest: 97, waist: 79, hip: 104, length: 106 }
    },
    fitProfile: 'Compressive Bodycon',
    drapeFactor: 0.98
  },
  {
    id: 'aline-gown',
    name: 'Aetheria Corset A-Line Gown',
    category: 'Formal Ballgown',
    fabric: 'Double-Faced Satin & Tulle Sub-layer',
    elasticity: 'Zero Stretch (Structured Corset)',
    description: 'Structured boned bodice with dramatic flared floor-length A-line skirt. High bust clearance required.',
    baseColor: '#9D4EDD',
    metallic: 0.3,
    roughness: 0.3,
    sizes: {
      XS: { chest: 82, waist: 62, hip: 95, length: 145 },
      S:  { chest: 86, waist: 66, hip: 100, length: 147 },
      M:  { chest: 90, waist: 70, hip: 105, length: 149 },
      L:  { chest: 95, waist: 75, hip: 110, length: 151 },
      XL: { chest: 102, waist: 82, hip: 116, length: 153 }
    },
    fitProfile: 'Cinched Waist / Flared Skirt',
    drapeFactor: 1.4
  },
  {
    id: 'wrap-shirt-dress',
    name: 'Kinetic Tailored Wrap Dress',
    category: 'Work & Casual Luxury',
    fabric: 'Eco-Viscose Twill',
    elasticity: 'Medium Adjustability',
    description: 'Versatile belted wrap dress featuring crisp lapels, adjustable waist tie, and asymmetric tulip hemline.',
    baseColor: '#10B981',
    metallic: 0.0,
    roughness: 0.6,
    sizes: {
      XS: { chest: 84, waist: 66, hip: 90, length: 105 },
      S:  { chest: 88, waist: 70, hip: 94, length: 107 },
      M:  { chest: 92, waist: 74, hip: 98, length: 109 },
      L:  { chest: 97, waist: 79, hip: 103, length: 111 },
      XL: { chest: 103, waist: 85, hip: 109, length: 113 }
    },
    fitProfile: 'Adjustable Wrap Fit',
    drapeFactor: 1.2
  }
];
